"""
src/evaluation/run_benchmark.py
================================
Run the full evaluation benchmark across scenarios S1-S6.

Usage:
    python -m src.evaluation.run_benchmark

Currently only S1 (originals-only baseline) is runnable since
synthetic data (S2-S6) hasn't been generated yet.
"""

import json
import os
import sys
import copy

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import load_config, build_dataloaders
from src.model.framework import MedicalDeepfakeDetector
from src.evaluation.benchmark import Evaluator

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "config.yaml"
)


def main():
    print("=" * 60)
    print("  Benchmark Evaluation Runner")
    print("=" * 60)

    cfg = load_config(CONFIG_PATH)
    cfg = copy.deepcopy(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    if device == "cpu":
        cfg["data"]["num_workers"] = 0

    # Check for test split
    test_csv = cfg["paths"]["test_csv"]
    if not os.path.exists(test_csv):
        print(f"\n[ERROR] Test CSV not found: {test_csv}")
        print("  Run:  python prepare_dev_subset.py")
        sys.exit(1)

    # Load model + checkpoint
    model = MedicalDeepfakeDetector(cfg).to(device)

    ckpt_dir = cfg["paths"].get("checkpoints", "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best_forgery_auc.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print("  [WARN] No checkpoint found - evaluating untrained model")

    # Build test DataLoader
    _, _, test_loader = build_dataloaders(cfg)

    # Run S3 evaluation (mixed sets)
    evaluator = Evaluator(model, cfg, device)
    results = evaluator.evaluate(test_loader, scenario_name="S3_mixed")

    # Save results
    reports_dir = cfg["paths"].get("reports", "outputs/reports")
    os.makedirs(reports_dir, exist_ok=True)
    results_path = os.path.join(reports_dir, "benchmark_s3.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
