import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

def compute_pos_weights(train_csv: str, num_classes: int = 14) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    pos_weights = []
    for cls in DISEASE_CLASSES:
        positives = df["diseases"].str.contains(cls, na=False).sum()
        negatives = len(df) - positives
        weight = min(negatives / max(positives, 1), 50.0)
        pos_weights.append(weight)
    return torch.tensor(pos_weights, dtype=torch.float32)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, 1, H, W), targets: (B, 1, H, W)
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0, gamma=0.3, pos_weight=None, forgery_pos_weight=10.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.bce_disease = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        # forgery_pos_weight handles the ~5:1 authentic:fake imbalance
        f_pw = torch.tensor([forgery_pos_weight])
        self.bce_forgery = nn.BCEWithLogitsLoss(pos_weight=f_pw)
        self.bce_localization = nn.BCEWithLogitsLoss()
        self.dice_localization = DiceLoss()
        
    def forward(self, preds, targets):
        """
        preds: dict returned by framework
        targets: dict containing 'disease_labels', 'forgery_labels', 'localization_masks'
        """
        # 1. Disease loss
        disease_loss = self.bce_disease(preds['disease_logits'], targets['disease_labels'].float())
        
        # 2. Forgery loss
        forgery_loss = self.bce_forgery(preds['forgery_logits'].squeeze(-1), targets['forgery_labels'].float())
        
        # 3. Localization loss (only calculated for manipulated images, or where mask is present)
        # targets['localization_masks'] shape: (B, 1, H, W)
        loc_logits = preds['localization_logits']
        loc_masks = targets['localization_masks'].float()
        
        # In a batch there might be authentic images without masks (all zeros). 
        # Dice loss should generally handle this, but to be clean, BCE + Dice.
        loc_bce = self.bce_localization(loc_logits, loc_masks)
        loc_dice = self.dice_localization(loc_logits, loc_masks)
        localization_loss = loc_bce + loc_dice
        
        # Combine
        total_loss = (self.alpha * disease_loss) + \
                     (self.beta * forgery_loss) + \
                     (self.gamma * localization_loss)
                     
        return {
            'total_loss': total_loss,
            'disease_loss': disease_loss,
            'forgery_loss': forgery_loss,
            'localization_loss': localization_loss
        }
