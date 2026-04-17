# ============================================================
# deepfake_pipeline/freq_perturb.py
# Forensic-Aware Medical Deepfake Detection System
# VAC - Healthcare Security Project | PRD v2.0
#
# SUBTLE SPECTRAL MANIPULATION via FFT PERTURBATION
# --------------------------------------------------
# Adds synthetic frequency-domain artifacts to authentic images
# to improve zero-day robustness of the FFT-CNN branch.
# ============================================================

import os
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


VALID_MODES = ("grid", "noise", "band")
VALID_BANDS = ("low", "mid", "high")


def generate_grid_artifact(
    fft_shifted: np.ndarray,
    intensity: float,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Simulate checkerboard-like GAN upsampling artifacts by injecting
    periodic spikes in frequency space.
    """
    if stride is None:
        stride = random.choice([8, 16, 32])

    h, w = fft_shifted.shape
    cy, cx = h // 2, w // 2
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)

    artifact = np.zeros((h, w), dtype=np.float32)
    base_amp = float(np.percentile(magnitude, 99))
    spike_amp = max(base_amp, 1.0) * (0.35 + 1.15 * intensity)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if abs(y - cy) < 2 and abs(x - cx) < 2:
                continue
            artifact[y, x] = spike_amp

    perturbed_magnitude = magnitude + artifact
    perturbed = perturbed_magnitude * np.exp(1j * phase)
    meta = {"stride": stride}
    return perturbed, meta


def generate_noise_artifact(
    fft_shifted: np.ndarray,
    intensity: float,
    sigma: float = 0.015,
) -> Tuple[np.ndarray, Dict]:
    """
    Add Gaussian complex noise in frequency space.
    """
    magnitude = np.abs(fft_shifted)
    scale = (sigma * max(intensity, 1e-6)) * float(np.mean(magnitude))
    noise_real = np.random.normal(0.0, scale, fft_shifted.shape)
    noise_imag = np.random.normal(0.0, scale, fft_shifted.shape)
    perturbed = fft_shifted + (noise_real + 1j * noise_imag)
    meta = {"sigma": sigma}
    return perturbed, meta


def generate_band_artifact(
    fft_shifted: np.ndarray,
    intensity: float,
    band: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Boost a ring in low/mid/high frequency range.
    """
    if band is None:
        band = random.choice(list(VALID_BANDS))
    if band not in VALID_BANDS:
        raise ValueError(f"Unsupported band '{band}'. Expected one of {VALID_BANDS}.")

    h, w = fft_shifted.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(min(cx, cy))

    if band == "low":
        r1, r2 = 0.08 * rmax, 0.22 * rmax
    elif band == "mid":
        r1, r2 = 0.30 * rmax, 0.55 * rmax
    else:
        r1, r2 = 0.68 * rmax, 0.95 * rmax

    mask = (rr >= r1) & (rr <= r2)
    boost_factor = 1.0 + intensity * 2.0

    perturbed = fft_shifted.copy()
    perturbed[mask] = perturbed[mask] * boost_factor
    meta = {"band": band, "boost_factor": round(boost_factor, 4)}
    return perturbed, meta


def add_frequency_perturbation(
    image_path: str,
    output_path: str,
    intensity: float,
    mode: str,
    sigma: float = 0.015,
    band: Optional[str] = None,
) -> Dict:
    """
    Apply frequency perturbation and save a raw perturbed grayscale image.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_MODES}.")

    intensity = float(np.clip(intensity, 0.0, 1.0))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_f = image.astype(np.float32)
    fft = np.fft.fft2(image_f)
    fft_shifted = np.fft.fftshift(fft)

    if mode == "grid":
        perturbed_shifted, mode_meta = generate_grid_artifact(fft_shifted, intensity=intensity)
    elif mode == "noise":
        perturbed_shifted, mode_meta = generate_noise_artifact(
            fft_shifted, intensity=intensity, sigma=sigma
        )
    else:
        perturbed_shifted, mode_meta = generate_band_artifact(
            fft_shifted, intensity=intensity, band=band
        )

    inv_shift = np.fft.ifftshift(perturbed_shifted)
    recon = np.fft.ifft2(inv_shift)
    recon_real = np.real(recon)
    recon_uint8 = np.clip(recon_real, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, recon_uint8)

    return {
        "image_path": output_path,
        "mode": mode,
        "intensity": round(float(intensity), 4),
        "mode_meta": mode_meta,
    }


def run_perturbation_pipeline(
    source_csv: str,
    output_dir: str,
    manifest_path: str,
    max_images: Optional[int] = None,
    seed: int = 42,
    modes: Tuple[str, ...] = VALID_MODES,
    subtle_fraction: float = 0.4,
    subtle_range: Tuple[float, float] = (0.05, 0.30),
    normal_range: Tuple[float, float] = (0.30, 0.85),
) -> None:
    """
    Batch perturbation runner compatible with existing split/manifest schema.
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    df = pd.read_csv(source_csv)
    if "is_manipulated" in df.columns:
        authentic_df = df[df["is_manipulated"].astype(str).str.lower().isin(["false", "0"])].copy()
        if len(authentic_df) == 0:
            authentic_df = df.copy()
    else:
        authentic_df = df.copy()

    authentic_df = authentic_df.reset_index(drop=True)
    if max_images is not None:
        authentic_df = authentic_df.head(max_images)

    print(f"[FreqPerturb] Source images: {len(authentic_df)}")
    print(f"[FreqPerturb] Modes: {list(modes)} | Subtle fraction target: {subtle_fraction}")

    manifest: List[Dict] = []
    success_count = 0
    fail_count = 0

    for i, row in authentic_df.iterrows():
        image_path = str(row.get("image_path", ""))
        source_id = str(row.get("image_id", i))
        diseases = str(row.get("diseases", ""))

        if not image_path or not os.path.exists(image_path):
            fail_count += 1
            continue

        mode = random.choice(modes)
        if random.random() < subtle_fraction:
            intensity = random.uniform(*subtle_range)
        else:
            intensity = random.uniform(*normal_range)

        try:
            out_name = f"{source_id}_freq_{mode}.png"
            out_path = os.path.join(output_dir, out_name)

            result = add_frequency_perturbation(
                image_path=image_path,
                output_path=out_path,
                intensity=intensity,
                mode=mode,
            )

            record = {
                "image_id": f"{source_id}_freq_{mode}",
                "image_path": result["image_path"],
                "source_image_id": source_id,
                "diseases": diseases,
                "is_manipulated": True,
                "manipulation_type": "subtle",
                "manipulation_mask_path": "",
                "manipulation_bbox": "",
                "manipulation_intensity": result["intensity"],
                "generator": "freq_perturb_v1",
                "perturbation_mode": result["mode"],
                "mode_meta": result["mode_meta"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            manifest.append(record)
            success_count += 1

            if success_count % 200 == 0:
                _save_manifest(manifest, manifest_path)
                print(f"[FreqPerturb] Progress: {success_count} generated...")

        except Exception as e:
            print(f"[FreqPerturb] ERROR on {source_id}: {e}")
            fail_count += 1

    _save_manifest(manifest, manifest_path)
    print("\n[FreqPerturb] Complete.")
    print(f"  Success: {success_count} | Failed: {fail_count}")
    print(f"  Manifest saved to: {manifest_path}")


def _save_manifest(manifest: List[Dict], manifest_path: str) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def test_single_image(
    image_path: str,
    mode: str,
    intensity: float,
    output_path: str = "outputs/test_freq_perturb.png",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = add_frequency_perturbation(
        image_path=image_path,
        output_path=output_path,
        intensity=intensity,
        mode=mode,
    )
    print(f"[FreqPerturb][Test] Saved: {result['image_path']}")
    print(f"[FreqPerturb][Test] Mode: {result['mode']}")
    print(f"[FreqPerturb][Test] Intensity: {result['intensity']}")
    print(f"[FreqPerturb][Test] Meta: {result['mode_meta']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency-domain perturbation pipeline")
    subparsers = parser.add_subparsers(dest="command")

    tp = subparsers.add_parser("test", help="Single-image test perturbation")
    tp.add_argument("--image", required=True)
    tp.add_argument("--mode", dest="perturb_mode", default="grid", choices=VALID_MODES)
    tp.add_argument("--intensity", type=float, default=0.4)
    tp.add_argument("--output", default="outputs/test_freq_perturb.png")

    bp = subparsers.add_parser("batch", help="Batch perturbation run")
    bp.add_argument("--source_csv", required=True)
    bp.add_argument("--output_dir", default="data/synthetic/subtle")
    bp.add_argument("--manifest_path", default="data/synthetic/subtle/freq_manifest.json")
    bp.add_argument("--max_images", type=int, default=None)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--subtle_fraction", type=float, default=0.4)

    args = parser.parse_args()

    if args.command == "test":
        test_single_image(
            image_path=args.image,
            mode=args.perturb_mode,
            intensity=args.intensity,
            output_path=args.output,
        )
    elif args.command == "batch":
        run_perturbation_pipeline(
            source_csv=args.source_csv,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            max_images=args.max_images,
            seed=args.seed,
            subtle_fraction=args.subtle_fraction,
        )
    else:
        parser.print_help()
