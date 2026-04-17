"""
src/evaluation/run_ablation.py
==============================
Ablation study runner (PRD v2, Section 8.3 D3.2).

Trains + evaluates each baseline variant:
    B1: DenseNet-121 classifier only (no forgery)
    B2: Separate backbones (no shared weights)
    B3: Full system without FFT branch
    B4: Full system (our model)

Uses the same dev subset and config. Results are written to
    outputs/reports/ablation_results.json

Usage:
    python -m src.evaluation.run_ablation
"""

import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import load_config, build_dataloaders
from src.model.framework import MedicalDeepfakeDetector
from src.model.baselines import build_baseline
from src.training.trainer import Trainer
from src.evaluation.benchmark import Evaluator

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "config.yaml"
)


def train_and_evaluate(model, cfg, device, model_name, ablation_epochs=5):
    """Train a model for a few epochs and evaluate on the test set."""
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["training"]["total_epochs"] = ablation_epochs

    if device == "cpu":
        cfg_copy["data"]["num_workers"] = 0

    train_loader, val_loader, test_loader = build_dataloaders(cfg_copy)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg_copy,
        device=device,
    )

    print(f"\n  Training {model_name} for {ablation_epochs} epochs ...")
    trainer.fit()

    print(f"\n  Evaluating {model_name} ...")
    evaluator = Evaluator(model, cfg_copy, device)
    metrics = evaluator.evaluate(test_loader, scenario_name=f"Ablation_{model_name}")

    return metrics


def main():
    print("=" * 60)
    print("  Ablation Study Runner")
    print("=" * 60)

    cfg = load_config(CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # Check splits exist
    if not os.path.exists(cfg["paths"]["train_csv"]):
        print("\n[ERROR] Split CSVs not found. Run: python prepare_dev_subset.py")
        sys.exit(1)

    # Ablation epochs (shorter than full training - just enough to compare)
    ABLATION_EPOCHS = 5

    all_results = {}

    # ── B1: Classifier Only ───────────────────────────────────
    print("\n" + "=" * 40)
    print("  B1: DenseNet-121 Classifier Only")
    print("=" * 40)
    cfg_b1 = copy.deepcopy(cfg)
    cfg_b1["model"]["backbone"]["pretrained"] = True
    model_b1 = build_baseline("B1", cfg_b1).to(device)
    all_results["B1_classifier_only"] = train_and_evaluate(
        model_b1, cfg_b1, device, "B1", ABLATION_EPOCHS
    )

    # ── B3: No FFT Branch ─────────────────────────────────────
    print("\n" + "=" * 40)
    print("  B3: Full System without FFT Branch")
    print("=" * 40)
    cfg_b3 = copy.deepcopy(cfg)
    cfg_b3["model"]["backbone"]["pretrained"] = True
    model_b3 = build_baseline("B3", cfg_b3).to(device)
    all_results["B3_no_fft"] = train_and_evaluate(
        model_b3, cfg_b3, device, "B3", ABLATION_EPOCHS
    )

    # ── B4: Full System (Ours) ────────────────────────────────
    print("\n" + "=" * 40)
    print("  B4: Full System (Ours)")
    print("=" * 40)
    cfg_b4 = copy.deepcopy(cfg)
    cfg_b4["model"]["backbone"]["pretrained"] = True
    model_b4 = MedicalDeepfakeDetector(cfg_b4).to(device)
    all_results["B4_full_system"] = train_and_evaluate(
        model_b4, cfg_b4, device, "B4", ABLATION_EPOCHS
    )

    # Note: B2 (separate backbones) is very expensive (2x DenseNet).
    # Skip by default; uncomment to include.
    # print("\n" + "=" * 40)
    # print("  B2: Separate Backbones (2x DenseNet)")
    # print("=" * 40)
    # cfg_b2 = copy.deepcopy(cfg)
    # cfg_b2["model"]["backbone"]["pretrained"] = True
    # model_b2 = build_baseline("B2", cfg_b2).to(device)
    # all_results["B2_separate_backbones"] = train_and_evaluate(
    #     model_b2, cfg_b2, device, "B2", ABLATION_EPOCHS
    # )

    # ── Save Results ──────────────────────────────────────────
    reports_dir = cfg["paths"].get("reports", "outputs/reports")
    os.makedirs(reports_dir, exist_ok=True)
    results_path = os.path.join(reports_dir, "ablation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary Table ─────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  ABLATION STUDY RESULTS")
    print("=" * 70)
    header = f"{'Model':<30} {'Disease AUC':>12} {'Forgery AUC':>12} {'IoU':>8}"
    print(header)
    print("-" * 70)
    for name, metrics in all_results.items():
        d_auc = metrics.get("disease_auc", float("nan"))
        f_auc = metrics.get("forgery_auc", float("nan"))
        iou = metrics.get("localization_iou", float("nan"))
        print(f"{name:<30} {d_auc:>12.4f} {f_auc:>12.4f} {iou:>8.4f}")
    print("=" * 70)
    print(f"\n  Full results: {results_path}")


if __name__ == "__main__":
    main()
