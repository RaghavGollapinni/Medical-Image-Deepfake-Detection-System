import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss, f1_score, confusion_matrix

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

def calculate_disease_metrics(preds, targets):
    """
    preds: (N, 14) probabilities
    targets: (N, 14) binary labels
    """
    aucs = []
    # Compute per-class AUC
    for i in range(targets.shape[1]):
        # Check if class is present in true labels (avoid value error)
        if len(np.unique(targets[:, i])) > 1:
            auc = roc_auc_score(targets[:, i], preds[:, i])
            aucs.append(auc)
        else:
            aucs.append(np.nan)
            
    mean_auc = np.nanmean(aucs)
    return {
        "mean_auc": mean_auc,
        "class_aucs": aucs
    }

def calculate_forgery_metrics(preds, targets):
    """
    preds: (N,) probabilities
    targets: (N,) binary labels (1=manipulated, 0=authentic)
    """
    if len(np.unique(targets)) > 1:
        auc = roc_auc_score(targets, preds)
    else:
        auc = np.nan
    # Threshold at 0.5
    preds_bin = (preds >= 0.5).astype(int)
    acc = accuracy_score(targets, preds_bin)
    
    return {
        "auc": auc,
        "accuracy": acc
    }

def calculate_localization_iou(preds, targets, threshold=0.5):
    """
    preds: (N, H, W) heatmap probabilities
    targets: (N, H, W) binary masks
    """
    preds_bin = preds >= threshold
    targets_bin = targets > 0.5
    
    intersection = np.logical_and(preds_bin, targets_bin).sum(axis=(1, 2))
    union = np.logical_or(preds_bin, targets_bin).sum(axis=(1, 2))
    
    # Avoid division by zero
    iou = np.where(union > 0, intersection / union, 1.0) # if union is 0 (no pred, no target => perfect)
    
    return float(np.mean(iou))

def calculate_trust_calibration_error(trust_scores, true_reliability, n_bins=10):
    """
    Expected Calibration Error (ECE) for Trust Scores
    trust_scores: (N,) float [0, 1] - predicted trust
    true_reliability: (N,) binary - 1 if diagnosis is reliable, 0 otherwise. 
        Reliable means (authentic) or (manipulated but non-overlapping).
        For simplicity, often true_reliability = 1 - is_manipulated.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(trust_scores, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(true_reliability[bin_idx])
            bin_conf = np.mean(trust_scores[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(bin_idx) / len(trust_scores))
            
    return ece

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, class_names: list = DISEASE_CLASSES) -> dict:
    y_bin = (y_pred >= threshold).astype(int)
    results = {}
    for i, cls in enumerate(class_names):
        try:
            results[cls] = {
                "precision": precision_score(y_true[:, i], y_bin[:, i], zero_division=0),
                "recall":    recall_score(y_true[:, i], y_bin[:, i], zero_division=0),
                "f1":        f1_score(y_true[:, i], y_bin[:, i], zero_division=0),
            }
        except Exception:
            pass
    try:
        results["macro"] = {
            "precision": precision_score(y_true, y_bin, average="macro", zero_division=0),
            "recall":    recall_score(y_true, y_bin, average="macro", zero_division=0),
            "f1":        f1_score(y_true, y_bin, average="macro", zero_division=0),
        }
    except Exception:
        pass
    return results

def compute_forgery_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    y_bin = (y_pred >= threshold).astype(int)
    return {
        "accuracy":  (y_bin == y_true).mean(),
        "precision": precision_score(y_true, y_bin, zero_division=0),
        "recall":    recall_score(y_true, y_bin, zero_division=0),
        "f1":        f1_score(y_true, y_bin, zero_division=0),
    }

def plot_forgery_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "outputs/reports/confusion_matrix.png", threshold: float = 0.5) -> None:
    y_bin = (y_pred >= threshold).astype(int)
    cm    = confusion_matrix(y_true, y_bin)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Authentic", "Manipulated"], yticklabels=["Authentic", "Manipulated"])
    plt.title("Forgery Detection — Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Confusion matrix saved: {save_path}")

def plot_training_curves(train_losses: list, val_losses: list, val_aucs: list, save_path: str = "outputs/reports/training_curves.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax1.plot(epochs, val_losses,   label="Val Loss",   marker="o")
    ax1.set_title("Training vs Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_aucs, label="Val Forgery AUC", marker="o", color="green")
    ax2.set_title("Validation Forgery AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Training curves saved: {save_path}")
