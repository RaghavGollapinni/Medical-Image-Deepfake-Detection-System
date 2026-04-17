"""
run_phase23.py
==============
Phase 2+3 training: full fine-tuning after Phase 1 warm-up.

Phase 2 (epochs 6-25): Unfreeze backbone, LR=1e-4, joint fine-tuning
Phase 3 (epochs 26-35): Low LR=1e-5, stabilization

Prerequisite:
    Phase 1 must have been run first (produces best_forgery_auc.pt checkpoint)

Run from project root:
    python -m src.training.run_phase23
"""

import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import load_config, build_dataloaders
from src.model.framework import MedicalDeepfakeDetector
from src.training.trainer import Trainer
from src.training.utils import set_seed

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "config.yaml"
)


def main():
    print("=" * 60)
    print("  Phase 2+3 Training - Full Fine-Tuning")
    print("=" * 60)

    cfg = load_config(CONFIG_PATH)
    cfg = copy.deepcopy(cfg)

    # Apply random seed
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # Use pretrained backbone
    cfg["model"]["backbone"]["pretrained"] = True

    # Keep full epochs (Phase 2 + 3)
    cfg["training"]["total_epochs"] = int(cfg["training"]["total_epochs"])

    # Windows-safe num_workers
    if device == "cpu":
        cfg["data"]["num_workers"] = 0

    os.environ.setdefault("WANDB_MODE", "offline")

    for key in ("logs", "checkpoints", "outputs"):
        os.makedirs(cfg["paths"].get(key, key), exist_ok=True)

    # Check split CSVs
    train_csv = cfg["paths"]["train_csv"]
    if not os.path.exists(train_csv):
        print("\n[ERROR] Split CSVs not found. Run: python prepare_dev_subset.py")
        sys.exit(1)

    # Build DataLoaders
    print("\n[Step 1/3] Building DataLoaders ...")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Build model
    print("\n[Step 2/3] Building model ...")
    model = MedicalDeepfakeDetector(cfg).to(device)

    # Resume from Phase 1 best checkpoint
    ckpt_dir = cfg["paths"].get("checkpoints", "checkpoints")
    phase1_ckpt = os.path.join(ckpt_dir, "best_forgery_auc.pt")
    if not os.path.exists(phase1_ckpt):
        phase1_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    if os.path.exists(phase1_ckpt):
        state = torch.load(phase1_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded Phase 1 checkpoint: {phase1_ckpt}")
    else:
        print("  [WARN] No Phase 1 checkpoint found - starting from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Train Phases 2+3
    start_epoch = int(cfg["training"]["phase1"]["epochs"]) + 1
    end_epoch = int(cfg["training"]["total_epochs"])
    print(f"\n[Step 3/3] Training epochs {start_epoch}-{end_epoch} ...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
    )

    history = trainer.fit(start_epoch=start_epoch, end_epoch=end_epoch)

    # Save history
    summary_path = os.path.join(cfg["paths"].get("outputs", "outputs"), "phase23_history.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Phase 2+3 complete.")
    print(f"  History  -> {summary_path}")
    print(f"  Best ckpt-> {ckpt_dir}")
    print("\nNext: python -m src.evaluation.run_benchmark")


if __name__ == "__main__":
    main()
