"""
src/inference/predict.py
========================
Single-image and batch inference with Verified Diagnosis JSON output.
Produces the output format specified in PRD v2 Section 3.1 FR-05.

Usage:
    from src.inference.predict import DeepfakePredictor

    predictor = DeepfakePredictor(config, "checkpoints/best_forgery_auc.pt")
    result, overlay = predictor.predict("path/to/xray.jpg", generate_heatmap=True)
    print(json.dumps(result, indent=2))
"""

import json
import os
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class DeepfakePredictor:
    """
    End-to-end predictor for the Forensic-Aware Medical Deepfake Detector.

    Loads a trained checkpoint and provides single-image inference that returns
    the full Verified Diagnosis JSON schema from the PRD.
    """

    def __init__(self, config: dict, model_checkpoint: str):
        self.config = config
        self.device = torch.device(
            config.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Load model ──────────────────────────────────────────
        from ..model.framework import MedicalDeepfakeDetector

        self.model = MedicalDeepfakeDetector(config)
        self.model.load_state_dict(
            torch.load(model_checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        # ── Preprocessing config ────────────────────────────────
        self.image_size = config["data"]["image_size"]
        self.clahe_cfg = config["data"]["clahe"]
        self.norm_mean = config["data"]["normalization"]["mean"]
        self.norm_std = config["data"]["normalization"]["std"]
        self.classes = config["model"]["classification_head"]["classes"]

        # ── Transforms (val-mode: no augmentation) ──────────────
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])

        # ── Trust score calculator ──────────────────────────────
        from .trust_score import TrustScoreCalculator
        self.ts_calculator = TrustScoreCalculator(config)

    def _preprocess(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load an X-ray image and apply the same preprocessing as the dataset pipeline.

        Returns:
            input_tensor: (1, 3, H, W) tensor ready for model
            img_gray_np:  original grayscale numpy for heatmap overlay
        """
        # Read as grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_gray_orig = img_gray.copy()

        # Resize
        img_gray = cv2.resize(img_gray, (self.image_size, self.image_size))

        # CLAHE
        if self.clahe_cfg.get("enabled", True):
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_cfg.get("clip_limit", 2.0),
                tileGridSize=(self.clahe_cfg.get("tile_size", 8),) * 2,
            )
            img_gray = clahe.apply(img_gray)

        # 3-channel (repeat grayscale for ImageNet pretrained backbone)
        img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.uint8)
        pil_img = Image.fromarray(img_rgb)

        # Apply transforms
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        return input_tensor, img_gray_orig

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        generate_heatmap: bool = False,
    ) -> Tuple[Dict, Optional[np.ndarray]]:
        """
        Run inference on a single chest X-ray image.

        Args:
            image_path:       Path to the X-ray image file
            generate_heatmap: Whether to generate a heatmap overlay

        Returns:
            result:  Verified Diagnosis JSON dict (PRD FR-05 schema)
            overlay: BGR numpy array with heatmap overlay, or None
        """
        input_tensor, img_gray_orig = self._preprocess(image_path)

        # ── Forward pass ────────────────────────────────────────
        preds = self.model(input_tensor)

        disease_probs = torch.sigmoid(preds["disease_logits"]).squeeze(0).cpu().numpy()
        forgery_prob = float(torch.sigmoid(preds["forgery_logits"]).item())
        loc_heatmap = torch.sigmoid(preds["localization_logits"]).squeeze(0).squeeze(0).cpu().numpy()

        # ── Disease predictions (sorted by confidence) ──────────
        disease_predictions = sorted(
            [{"class": c, "prob": round(float(disease_probs[i]), 4)} for i, c in enumerate(self.classes)],
            key=lambda x: x["prob"],
            reverse=True,
        )
        top_disease = disease_predictions[0]

        # ── Trust score (overlap-aware) ─────────────────────────
        # Overlap = spatial correlation between localization heatmap
        # and disease-relevant regions. For now, use heatmap intensity
        # as a proxy for overlap.
        overlap = float(np.mean(loc_heatmap)) if forgery_prob > 0.5 else 0.0
        trust_score = self.ts_calculator.compute_score(forgery_prob, overlap=overlap)
        recommendation = self.ts_calculator.get_recommendation(trust_score)

        is_genuine = forgery_prob < 0.5

        # ── Heatmap overlay ─────────────────────────────────────
        overlay = None
        if generate_heatmap:
            from .visualize import create_overlay
            img_bgr = cv2.cvtColor(img_gray_orig, cv2.COLOR_GRAY2BGR)
            overlay = create_overlay(img_bgr, loc_heatmap, self.config)

        # ── Verified Diagnosis JSON (PRD FR-05) ─────────────────
        result = {
            "diagnosis": top_disease["class"],
            "primary_class": top_disease["class"],
            "confidence": top_disease["prob"],
            "is_genuine": is_genuine,
            "manipulation_prob": round(forgery_prob, 4),
            "trust_score": round(trust_score, 4),
            "disease_predictions": disease_predictions,
            "suspicious_regions": "See heatmap overlay" if (generate_heatmap and not is_genuine) else [],
            "recommendation": recommendation,
        }

        return result, overlay

    def predict_batch(
        self, image_paths: List[str], generate_heatmaps: bool = False
    ) -> List[Tuple[Dict, Optional[np.ndarray]]]:
        """Run inference on multiple images sequentially."""
        results = []
        for path in image_paths:
            try:
                result, overlay = self.predict(path, generate_heatmap=generate_heatmaps)
                results.append((result, overlay))
            except Exception as e:
                results.append(({"error": str(e), "image_path": path}, None))
        return results
