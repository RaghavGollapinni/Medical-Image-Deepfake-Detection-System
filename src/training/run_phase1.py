"""
run_phase1.py
=============
Phase 1 training: heads-only warm-up with backbone frozen.

Run from the project root:
    python -m src.training.run_phase1

Or directly:
    python src/training/run_phase1.py

What this does:
  - Loads config.yaml
  - Builds DataLoaders from the dev-subset split CSVs
    (run 'python prepare_dev_subset.py' first if splits don't exist)
  - Instantiates MedicalDeepfakeDetector with ImageNet-pretrained DenseNet-121
  - Trains for Phase 1 epochs (backbone frozen, heads only)
  - Saves best checkpoint to checkpoints/best_forgery_auc.pt
  - Saves training history JSON to outputs/phase1_history.json
"""

import json
import os
import sys
import copy

import torch

# Allow running directly as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import load_config, build_dataloaders
from src.model.framework import MedicalDeepfakeDetector
from src.training.trainer import Trainer
from src.training.utils import set_seed

# ── Config path (relative to project root or absolute) ────────
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "config.yaml"
)


def main():
    print("=" * 60)
    print("  Phase 1 Training — Backbone Frozen, Heads Warm-up")
    print("=" * 60)

    # Load config
    cfg = load_config(CONFIG_PATH)
    cfg = copy.deepcopy(cfg)

    # Apply random seed
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # ── Check split CSVs exist ─────────────────────────────────
    train_csv = cfg["paths"]["train_csv"]
    if not os.path.exists(train_csv):
        print("\n[ERROR] Split CSVs not found.")
        print("  Run this first:  python prepare_dev_subset.py")
        print(f"  Expected:        {train_csv}")
        sys.exit(1)

    # ── Device setup ───────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device    : {device}")
    if device == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Apply phase-1-safe overrides ──────────────────────────
    # Use pretrained weights (ImageNet — CheXNet weights would be better but
    # require separate download; ImageNet is sufficient for Phase 1 warm-up)
    cfg["model"]["backbone"]["pretrained"] = True

    # Limit total_epochs to Phase 1 only
    p1_epochs = int(cfg["training"]["phase1"]["epochs"])
    cfg["training"]["total_epochs"] = p1_epochs

    # Windows-safe: num_workers set in config.yaml (2). Override to 0 if
    # running on CPU to avoid multiprocessing overhead.
    if device == "cpu":
        cfg["data"]["num_workers"] = 0
        print("  Note      : CPU detected — num_workers set to 0")

    # Disable W&B for a clean local run (still writes TensorBoard logs)
    os.environ.setdefault("WANDB_MODE", "offline")

    # Ensure output directories exist
    for key in ("logs", "checkpoints", "outputs"):
        os.makedirs(cfg["paths"].get(key, key), exist_ok=True)

    # ── Build DataLoaders ──────────────────────────────────────
    print("\n[Step 1/3] Building DataLoaders …")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # ── Build model ────────────────────────────────────────────
    print("\n[Step 2/3] Building model …")
    model = MedicalDeepfakeDetector(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}  (backbone will be frozen by Trainer)")

    # ── Train ──────────────────────────────────────────────────
    print(f"\n[Step 3/3] Training Phase 1 ({p1_epochs} epoch(s)) …\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        # No max_train_batches / max_val_batches — use full dataset
    )

    history = trainer.fit()

    # ── Save history ───────────────────────────────────────────
    summary_path = os.path.join(
        cfg["paths"].get("outputs", "outputs"),
        "phase1_history.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Phase 1 complete.")
    print(f"  History  → {summary_path}")
    print(f"  Best ckpt→ {cfg['paths']['checkpoints']}best_forgery_auc.pt")
    print("\nNext step:  python -m src.training.run_phase23")


if __name__ == "__main__":
    main()
