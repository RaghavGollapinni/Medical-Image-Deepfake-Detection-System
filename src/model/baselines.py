"""
src/model/baselines.py
======================
Baseline model variants for ablation comparison (PRD v2, Section 7.2).

B1: DenseNet-121 classifier only (no forgery head, no FFT)
B2: Separate classifier + separate forgery detector (no shared backbone)
B3: Full system without FFT branch (spatial-only forgery detection)
B4: Full system (= MedicalDeepfakeDetector in framework.py)
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .backbone import SharedDenseNet
from .heads import DiseaseClassificationHead, ForgeryDetectionHead, LocalizationHead


# ─── B1: Disease Classification Only ─────────────────────────
class BaselineB1_ClassifierOnly(nn.Module):
    """
    B1: Standard DenseNet-121 disease classifier.
    No forgery detection, no FFT, no localization.
    Purpose: reference for diagnosis-only accuracy.
    """

    def __init__(self, config):
        super().__init__()
        pretrained = config["model"]["backbone"]["pretrained"]
        cls_hidden = config["model"]["classification_head"]["hidden_dim"]
        num_classes = config["model"]["classification_head"]["num_classes"]
        dropout = config["model"]["classification_head"]["dropout"]

        self.backbone = SharedDenseNet(pretrained=pretrained)
        self.disease_head = DiseaseClassificationHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=cls_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        pooled, features = self.backbone(x)
        disease_logits = self.disease_head(pooled)

        # Return compatible dict — forgery/localization are dummy zeros
        B = x.size(0)
        H = x.size(2)
        return {
            "disease_logits": disease_logits,
            "forgery_logits": torch.zeros(B, 1, device=x.device),
            "localization_logits": torch.zeros(B, 1, H, H, device=x.device),
        }


# ─── B2: Separate Classifier + Forgery (No Shared Weights) ───
class BaselineB2_SeparateModels(nn.Module):
    """
    B2: Two independent DenseNet-121 backbones — one for disease
    classification, one for forgery detection.
    No shared features. Measures benefit of multi-task learning.
    """

    def __init__(self, config):
        super().__init__()
        pretrained = config["model"]["backbone"]["pretrained"]
        cls_hidden = config["model"]["classification_head"]["hidden_dim"]
        num_classes = config["model"]["classification_head"]["num_classes"]
        forgery_hidden = config["model"]["forgery_head"]["hidden_dim"]
        dropout = config["model"]["classification_head"]["dropout"]

        # Separate backbone for disease
        self.disease_backbone = SharedDenseNet(pretrained=pretrained)
        self.disease_head = DiseaseClassificationHead(
            input_dim=self.disease_backbone.output_dim,
            hidden_dim=cls_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Separate backbone for forgery
        self.forgery_backbone = SharedDenseNet(pretrained=pretrained)
        backbone_dim = self.forgery_backbone.output_dim
        self.forgery_head = ForgeryDetectionHead(
            input_dim=backbone_dim,  # no FFT concatenation
            hidden_dim=forgery_hidden,
            dropout=dropout,
        )

        self.localization_head = LocalizationHead()

    def forward(self, x):
        # Disease path
        d_pooled, _ = self.disease_backbone(x)
        disease_logits = self.disease_head(d_pooled)

        # Forgery path (independent backbone)
        f_pooled, f_features = self.forgery_backbone(x)
        forgery_logits = self.forgery_head(f_pooled)
        localization_logits = self.localization_head(f_features)

        return {
            "disease_logits": disease_logits,
            "forgery_logits": forgery_logits,
            "localization_logits": localization_logits,
        }


# ─── B3: Shared Backbone, No FFT Branch ──────────────────────
class BaselineB3_NoFFT(nn.Module):
    """
    B3: Shared DenseNet-121 + all 3 heads, but NO FFT-CNN branch.
    Forgery head receives only spatial features (1024-dim).
    Purpose: isolate the FFT branch contribution.
    """

    def __init__(self, config):
        super().__init__()
        pretrained = config["model"]["backbone"]["pretrained"]
        cls_hidden = config["model"]["classification_head"]["hidden_dim"]
        num_classes = config["model"]["classification_head"]["num_classes"]
        forgery_hidden = config["model"]["forgery_head"]["hidden_dim"]
        dropout = config["model"]["classification_head"]["dropout"]

        self.backbone = SharedDenseNet(pretrained=pretrained)
        backbone_dim = self.backbone.output_dim

        self.disease_head = DiseaseClassificationHead(
            input_dim=backbone_dim,
            hidden_dim=cls_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Forgery head takes ONLY spatial features (no FFT concat)
        self.forgery_head = ForgeryDetectionHead(
            input_dim=backbone_dim,
            hidden_dim=forgery_hidden,
            dropout=dropout,
        )

        self.localization_head = LocalizationHead()

    def forward(self, x):
        pooled, features = self.backbone(x)
        disease_logits = self.disease_head(pooled)
        forgery_logits = self.forgery_head(pooled)
        localization_logits = self.localization_head(features)

        return {
            "disease_logits": disease_logits,
            "forgery_logits": forgery_logits,
            "localization_logits": localization_logits,
        }


# ─── Factory ─────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "B1": BaselineB1_ClassifierOnly,
    "B2": BaselineB2_SeparateModels,
    "B3": BaselineB3_NoFFT,
}


def build_baseline(baseline_name: str, config: dict) -> nn.Module:
    """
    Factory to build a baseline model by name.

    Args:
        baseline_name: "B1", "B2", or "B3"
        config: config.yaml dict
    """
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{baseline_name}'. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[baseline_name](config)
