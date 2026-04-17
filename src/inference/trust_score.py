"""
src/inference/trust_score.py
============================
Overlap-aware Trust Score computation (PRD v2, Section 5.5).

Formula:
    trust = 1.0 - p_manip * (base_weight + overlap_weight * overlap)

Where:
    p_manip  = forgery detection probability
    overlap  = spatial overlap between localization heatmap and disease-relevant regions
               (0.0 = no diagnostic overlap, 1.0 = full overlap)

Thresholds (from config):
    >= 0.85  -> "Diagnosis Verified"
    >= 0.70  -> "Proceed with Caution"
    <  0.70  -> "Manual Review Recommended / Rescan"
"""

import numpy as np


class TrustScoreCalculator:
    def __init__(self, config: dict):
        ts_cfg = config.get("trust_score", {})
        self.base_weight = ts_cfg.get("base_weight", 0.5)
        self.overlap_weight = ts_cfg.get("overlap_weight", 0.5)
        self.thresholds = ts_cfg.get("thresholds", {})
        self.labels = ts_cfg.get("labels", {})

    def compute_score(self, manipulation_prob: float, overlap: float = 1.0) -> float:
        """
        Compute the trust score.

        Args:
            manipulation_prob: P(manipulated) from forgery head, in [0, 1]
            overlap:           spatial overlap factor (0=no diagnostic overlap,
                               1=manipulation fully overlaps pathology region).
                               Defaults to 1.0 (worst-case assumption).

        Returns:
            Trust score in [0.0, 1.0]. Higher = more trustworthy.
        """
        penalty = manipulation_prob * (self.base_weight + self.overlap_weight * overlap)
        return float(max(0.0, min(1.0, 1.0 - penalty)))

    def compute_batch(
        self,
        manipulation_probs: np.ndarray,
        overlaps: np.ndarray = None,
    ) -> np.ndarray:
        """Vectorised trust score for a batch of predictions."""
        if overlaps is None:
            overlaps = np.ones_like(manipulation_probs)
        penalties = manipulation_probs * (self.base_weight + self.overlap_weight * overlaps)
        return np.clip(1.0 - penalties, 0.0, 1.0)

    def get_label(self, trust_score: float) -> str:
        """Return the categorical trust label for a numeric score."""
        t_verified = self.thresholds.get("verified", 0.85)
        t_caution = self.thresholds.get("caution", 0.70)

        if trust_score >= t_verified:
            return self.labels.get("verified", "Diagnosis Verified")
        elif trust_score >= t_caution:
            return self.labels.get("caution", "Proceed with Caution")
        else:
            return self.labels.get("review", "Manual Review Recommended / Rescan")

    # Backwards-compatible alias
    def get_recommendation(self, trust_score: float) -> str:
        return self.get_label(trust_score)
