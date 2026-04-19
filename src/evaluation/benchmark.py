import torch
import numpy as np
from tqdm import tqdm

from .metrics import (
    calculate_disease_metrics,
    calculate_forgery_metrics,
    calculate_localization_iou,
    calculate_trust_calibration_error,
    compute_classification_metrics,
    compute_forgery_metrics as precision_forgery_metrics,
    plot_forgery_confusion_matrix,
    find_optimal_thresholds
)
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.ts_calculator = None
        self.disease_thresholds = 0.5
        self.forgery_threshold = 0.5
        
        from ..inference.trust_score import TrustScoreCalculator
        self.ts_calculator = TrustScoreCalculator(config)
        
    @torch.no_grad()
    def calibrate_thresholds(self, val_loader):
        print("Calibrating disease thresholds on validation set...")
        self.model.eval()
        all_d_preds, all_d_targets = [], []
        all_f_preds, all_f_targets = [], []
        for batch in tqdm(val_loader, desc="Calibration"):
            images = batch['image'].to(self.device)
            targets = batch['disease'].cpu().numpy()
            f_targets = batch['forgery'].cpu().numpy()
            preds = self.model(images)
            d_preds = torch.sigmoid(preds['disease_logits']).cpu().numpy()
            f_preds = torch.sigmoid(preds['forgery_logits']).squeeze(-1).cpu().numpy()
            all_d_preds.append(d_preds)
            all_d_targets.append(targets)
            all_f_preds.append(f_preds)
            all_f_targets.append(f_targets)

        d_preds = np.concatenate(all_d_preds, axis=0)
        d_targets = np.concatenate(all_d_targets, axis=0)
        self.disease_thresholds = find_optimal_thresholds(d_targets, d_preds)
        rounded_th = [round(t, 2) for t in self.disease_thresholds]
        print(f"Optimal disease thresholds computed: {rounded_th}")

        # Calibrate forgery threshold
        f_preds = np.concatenate(all_f_preds, axis=0)
        f_targets = np.concatenate(all_f_targets, axis=0)
        self.forgery_threshold = self._find_optimal_forgery_threshold(f_preds, f_targets)
        print(f"Optimal forgery threshold computed: {self.forgery_threshold:.4f}")

    def _find_optimal_forgery_threshold(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Find optimal threshold for forgery detection using F1 score."""
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.95, 0.01):
            y_bin = (y_pred >= t).astype(int)
            score = f1_score(y_true, y_bin, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        return best_t

    @torch.no_grad()
    def evaluate(self, dataloader, scenario_name="Standard"):
        self.model.eval()
        
        all_disease_preds = []
        all_disease_targets = []
        all_forgery_preds = []
        all_forgery_targets = []
        all_loc_preds = []
        all_loc_targets = []
        
        print(f"Running evaluation for scenario: {scenario_name}")
        for batch in tqdm(dataloader, desc="Scoring"):
            images = batch['image'].to(self.device)
            d_targets = batch['disease'].cpu().numpy()
            f_targets = batch['forgery'].cpu().numpy()
            l_targets = batch['mask'].squeeze(1).cpu().numpy()
            
            preds = self.model(images)
            
            d_preds = torch.sigmoid(preds['disease_logits']).cpu().numpy()
            f_preds = torch.sigmoid(preds['forgery_logits']).squeeze(-1).cpu().numpy()
            l_preds = torch.sigmoid(preds['localization_logits']).squeeze(1).cpu().numpy()
            
            all_disease_preds.append(d_preds)
            all_disease_targets.append(d_targets)
            all_forgery_preds.append(f_preds)
            all_forgery_targets.append(f_targets)
            all_loc_preds.append(l_preds)
            all_loc_targets.append(l_targets)
            
        # Concatenate
        d_preds = np.concatenate(all_disease_preds, axis=0)
        d_targets = np.concatenate(all_disease_targets, axis=0)
        f_preds = np.concatenate(all_forgery_preds, axis=0)
        f_targets = np.concatenate(all_forgery_targets, axis=0)
        l_preds = np.concatenate(all_loc_preds, axis=0)
        l_targets = np.concatenate(all_loc_targets, axis=0)
        
        # 1. Disease Metrics
        disease_results = calculate_disease_metrics(d_preds, d_targets)
        
        # 2. Forgery Metrics (using calibrated threshold)
        forgery_results = calculate_forgery_metrics(f_preds, f_targets, threshold=self.forgery_threshold)
        
        # 3. Localization Metrics
        iou = calculate_localization_iou(l_preds, l_targets, threshold=self.config.get('evaluation', {}).get('iou_threshold', 0.5))
        
        # 4. Trust Score Calibration
        # trust = 1.0 - forgery_prob (simplification from formula)
        trust_scores = np.array([self.ts_calculator.compute_score(p) for p in f_preds])
        # True reliability: 1 if authentic (not forged), 0 if forged
        true_reliability = 1 - f_targets
        
        ece = calculate_trust_calibration_error(trust_scores, true_reliability)
        
        # Alignment (Accuracy of Trust Score vs True Reliability classes)
        t_verified = self.config.get('trust_score', {}).get('thresholds', {}).get('verified', 0.85)
        # Assuming reliable -> > t_verified
        predicted_reliable = (trust_scores >= t_verified).astype(int)
        trust_alignment = np.mean(predicted_reliable == true_reliability)
        
        # 5. Advanced Metrics (Precision, Recall, F1)
        disease_results_ext = compute_classification_metrics(d_targets, d_preds, threshold=self.disease_thresholds)
        forgery_results_ext = precision_forgery_metrics(f_targets, f_preds, threshold=self.forgery_threshold)
        
        reports_dir = self.config.get('paths', {}).get('reports', 'outputs/reports/')
        import os
        os.makedirs(reports_dir, exist_ok=True)
        plot_forgery_confusion_matrix(f_targets, f_preds, save_path=os.path.join(reports_dir, "confusion_matrix.png"), threshold=self.forgery_threshold)
        
        metrics = {
            "disease_auc": disease_results["mean_auc"],
            "forgery_auc": forgery_results["auc"],
            "forgery_accuracy": forgery_results["accuracy"],
            "localization_iou": iou,
            "trust_ece": ece,
            "trust_alignment": trust_alignment,
            "forgery_f1": forgery_results_ext.get("f1", 0.0),
            "forgery_precision": forgery_results_ext.get("precision", 0.0),
            "forgery_recall": forgery_results_ext.get("recall", 0.0),
            "disease_macro_f1": disease_results_ext.get("macro", {}).get("f1", 0.0)
        }
        
        self.print_results(scenario_name, metrics)
        return metrics

    def print_results(self, scenario, metrics):
        print(f"\n--- Results for {scenario} ---")
        for key, val in metrics.items():
            print(f"{key}: {val:.4f}")
            
        targets = self.config.get('evaluation', {}).get('targets', {})
        print("\n--- PRD Target Validation ---")
        if targets:
            print(f"Disease AUC: {metrics['disease_auc']:.4f} (Target: {targets.get('diagnosis_auc', 0.85)}) "
                  f"[{'PASS' if metrics['disease_auc'] >= targets.get('diagnosis_auc', 0.85) else 'FAIL'}]")
            print(f"Forgery AUC: {metrics['forgery_auc']:.4f} (Target: {targets.get('forgery_auc', 0.92)}) "
                  f"[{'PASS' if metrics['forgery_auc'] >= targets.get('forgery_auc', 0.92) else 'FAIL'}]")
            print(f"Localization IoU: {metrics['localization_iou']:.4f} (Target: {targets.get('localization_iou', 0.50)}) "
                  f"[{'PASS' if metrics['localization_iou'] >= targets.get('localization_iou', 0.5) else 'FAIL'}]")
            print(f"Trust ECE: {metrics['trust_ece']:.4f} (Target: <= {targets.get('trust_ece', 0.10)}) "
                  f"[{'PASS' if metrics['trust_ece'] <= targets.get('trust_ece', 0.10) else 'FAIL'}]")
