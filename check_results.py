import os, json

ckpt = r"D:\Projects and Research\VAC - Healthcare Security\medical_deepfake_detector\checkpoints\best_forgery_auc.pt"
hist = r"D:\Projects and Research\VAC - Healthcare Security\medical_deepfake_detector\outputs\phase1_history.json"

print("Checkpoint:", "EXISTS" if os.path.exists(ckpt) else "MISSING",
      f"({os.path.getsize(ckpt)/1e6:.1f} MB)" if os.path.exists(ckpt) else "")

if os.path.exists(hist):
    with open(hist) as f:
        h = json.load(f)
    print(f"History: {len(h)} epochs saved")
    for e in h:
        d = e["train"]["disease_loss"]
        v = e["val"]["total_loss"]
        ep = e["epoch"]
        print(f"  Epoch {ep}: train_disease={d:.4f} val_total={v:.4f}")
else:
    print("History: MISSING")
