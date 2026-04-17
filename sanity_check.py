import sys, os, copy
sys.path.insert(0, '.')
from src.data.dataset import load_config, build_dataloaders
from src.model.framework import MedicalDeepfakeDetector
import torch

cfg = load_config('configs/config.yaml')
cfg = copy.deepcopy(cfg)
cfg['data']['num_workers'] = 0
cfg['model']['backbone']['pretrained'] = False  # skip download for fast check
cfg['training']['total_epochs'] = 1

print('Building dataloaders...')
train_loader, val_loader, test_loader = build_dataloaders(cfg)
print(f'  Train batches: {len(train_loader)}')
print(f'  Val   batches: {len(val_loader)}')
print(f'  Test  batches: {len(test_loader)}')

print('\nLoading one batch...')
batch = next(iter(train_loader))
print(f'  image shape  : {batch["image"].shape}')
print(f'  disease shape: {batch["disease"].shape}')
print(f'  forgery shape: {batch["forgery"].shape}')
print(f'  mask shape   : {batch["mask"].shape}')
print(f'  sample id    : {batch["metadata"]["image_id"][0]}')

print('\nBuilding model (no pretrained)...')
model = MedicalDeepfakeDetector(cfg)
model.eval()

with torch.no_grad():
    out = model(batch['image'])

print(f'  disease_logits     : {out["disease_logits"].shape}')
print(f'  forgery_logits     : {out["forgery_logits"].shape}')
print(f'  localization_logits: {out["localization_logits"].shape}')
print()
print('SANITY CHECK PASSED - pipeline is ready for training!')
