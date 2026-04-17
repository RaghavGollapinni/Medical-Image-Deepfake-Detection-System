# ============================================================
# deepfake_pipeline/roentgen_inject.py
# Forensic-Aware Medical Deepfake Detection System
# VAC - Healthcare Security Project | PRD v2.0
#
# INJECTION ATTACKS via RoentGen Inpainting
# ------------------------------------------
# Takes a HEALTHY chest X-ray and injects a synthetic pathology
# (nodule, lesion, mass, consolidation) into a specified region
# using RoentGen - a domain-adapted Stable Diffusion model
# fine-tuned on chest X-ray datasets.
#
# RoentGen paper: Chambon et al., arXiv:2211.12737 (2022)
# HuggingFace:    StanfordAIMI/radi-mag-chest-x-ray-2
# ============================================================

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# --- INJECTION PROMPTS --------------------------------------

INJECTION_PROMPTS = {
    "nodule": [
        "chest x-ray showing a solitary pulmonary nodule, well-defined round opacity",
        "frontal chest radiograph with a small pulmonary nodule in the right lower lobe",
        "chest x-ray with a calcified pulmonary nodule, benign appearance",
    ],
    "mass": [
        "chest x-ray showing a large pulmonary mass with irregular borders, suspicious for malignancy",
        "frontal chest radiograph with a hilar mass causing bronchial obstruction",
        "chest x-ray with a spiculated pulmonary mass in the left upper lobe",
    ],
    "lesion": [
        "chest x-ray showing focal consolidation and ground glass opacity",
        "frontal chest radiograph with a cavitary lesion and air-fluid level",
        "chest x-ray with a peripheral lung lesion adjacent to the pleura",
    ],
    "consolidation": [
        "chest x-ray showing lobar consolidation consistent with pneumonia",
        "frontal chest radiograph with right lower lobe consolidation and air bronchograms",
        "chest x-ray showing patchy bilateral consolidation",
    ],
    "infiltration": [
        "chest x-ray showing interstitial infiltrates bilaterally",
        "frontal chest radiograph with reticular infiltrates in the lower lobes",
        "chest x-ray with peribronchial infiltrates and bronchial wall thickening",
    ],
}

NEGATIVE_PROMPT = (
    "blurry, low quality, artifacts, distorted anatomy, "
    "unnatural texture, cartoon, painting, watermark"
)


# --- MASK GENERATORS ----------------------------------------

def generate_injection_mask(
    image_size: int,
    region: str = "random",
    bbox: Optional[List[int]] = None,
    pathology_type: str = "nodule",
) -> Tuple[np.ndarray, List[int]]:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    if bbox is not None:
        x, y, w, h = bbox
    else:
        x, y, w, h = _sample_region(image_size, region, pathology_type)

    cx, cy = x + w // 2, y + h // 2

    if pathology_type in ["nodule"]:
        radius = min(w, h) // 2
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    elif pathology_type in ["mass", "lesion"]:
        axes = (w // 2, h // 2)
        angle = random.randint(-30, 30)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    elif pathology_type in ["consolidation", "infiltration"]:
        num_pts = random.randint(5, 8)
        angles = sorted(random.uniform(0, 2 * np.pi) for _ in range(num_pts))
        pts = []
        for a in angles:
            rx = random.uniform(0.6, 1.0) * w // 2
            ry = random.uniform(0.6, 1.0) * h // 2
            pts.append([int(cx + rx * np.cos(a)), int(cy + ry * np.sin(a))])
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_np], 255)
    else:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask, [x, y, w, h]


def _sample_region(
    image_size: int,
    region: str,
    pathology_type: str,
) -> Tuple[int, int, int, int]:
    size_ranges = {
        "nodule": (0.04, 0.10),
        "mass": (0.12, 0.22),
        "lesion": (0.08, 0.18),
        "consolidation": (0.20, 0.35),
        "infiltration": (0.18, 0.30),
    }
    lo, hi = size_ranges.get(pathology_type, (0.08, 0.20))
    w = int(random.uniform(lo, hi) * image_size)
    h = int(random.uniform(lo, hi) * image_size)

    half = image_size // 2
    pad = int(0.10 * image_size)

    region_bounds = {
        "upper_left": (pad, pad, half - pad, half - pad),
        "upper_right": (half + pad, pad, image_size - pad - w, half - pad),
        "lower_left": (pad, half + pad, half - pad, image_size - pad - h),
        "lower_right": (half + pad, half + pad, image_size - pad - w, image_size - pad - h),
        "central": (half // 2, half // 2, half, half),
    }

    if region == "random":
        region = random.choice(list(region_bounds.keys()))

    x_min, y_min, x_max, y_max = region_bounds[region]
    x = random.randint(x_min, max(x_min, x_max - w))
    y = random.randint(y_min, max(y_min, y_max - h))
    return x, y, w, h


# --- PIPELINE LOADER ----------------------------------------

def load_roentgen_pipeline(device: str = "cuda") -> StableDiffusionInpaintPipeline:
    model_id = "StanfordAIMI/radi-mag-chest-x-ray-2"

    print(f"[RoentGen] Loading inpainting pipeline: {model_id}")
    print(f"[RoentGen] Device: {device}")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    print("[RoentGen] Pipeline loaded successfully.")
    return pipe


# --- CORE INJECTION FUNCTION --------------------------------

def inject_pathology(
    pipe: StableDiffusionInpaintPipeline,
    image_path: str,
    pathology_type: str = "nodule",
    region: str = "random",
    bbox: Optional[List[int]] = None,
    intensity: Optional[float] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    output_size: int = 224,
) -> Dict:
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
        seed = random.randint(0, 999999)

    source_img = Image.open(image_path).convert("RGB")
    source_img = source_img.resize((output_size, output_size), Image.LANCZOS)

    mask_np, final_bbox = generate_injection_mask(
        image_size=output_size,
        region=region,
        bbox=bbox,
        pathology_type=pathology_type,
    )
    mask_pil = Image.fromarray(mask_np).convert("RGB")

    prompt = random.choice(INJECTION_PROMPTS[pathology_type])

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=source_img,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    injected_image = result.images[0]

    if intensity is None:
        mask_coverage = (mask_np > 127).sum() / (output_size * output_size)
        intensity = float(np.clip(mask_coverage * 5.0, 0.1, 1.0))

    metadata = {
        "source_image": image_path,
        "pathology_type": pathology_type,
        "region": region,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "bbox": final_bbox,
        "intensity": round(intensity, 4),
        "generator": "roentgen_v1",
        "manipulation_type": "injection",
        "timestamp": datetime.utcnow().isoformat(),
    }

    return {
        "injected_image": injected_image,
        "mask": mask_np,
        "bbox": final_bbox,
        "prompt": prompt,
        "intensity": intensity,
        "metadata": metadata,
    }


# --- BATCH INJECTION PIPELINE -------------------------------

def run_injection_pipeline(
    source_csv: str,
    output_dir: str,
    masks_dir: str,
    manifest_path: str,
    pathology_types: Optional[List[str]] = None,
    n_per_image: int = 1,
    device: str = "cuda",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    output_size: int = 224,
    max_images: Optional[int] = None,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if pathology_types is None:
        pathology_types = list(INJECTION_PROMPTS.keys())

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    df = pd.read_csv(source_csv)
    healthy_df = df[
        df["diseases"].isna()
        | (df["diseases"] == "")
        | (df["diseases"].str.lower() == "no finding")
    ].reset_index(drop=True)

    if max_images is not None:
        healthy_df = healthy_df.head(max_images)

    print(f"[Injection Pipeline] Source healthy images: {len(healthy_df)}")
    print(f"[Injection Pipeline] Injections per image:  {n_per_image}")
    print(f"[Injection Pipeline] Pathology types:       {pathology_types}")
    print(f"[Injection Pipeline] Total target:          {len(healthy_df) * n_per_image}")

    pipe = load_roentgen_pipeline(device=device)

    manifest = []
    success_count = 0
    fail_count = 0

    for i, row in healthy_df.iterrows():
        image_path = row["image_path"]
        source_id = str(row["image_id"])

        if not os.path.exists(image_path):
            print(f"[Injection Pipeline] SKIP (not found): {image_path}")
            fail_count += 1
            continue

        for j in range(n_per_image):
            pathology = random.choice(pathology_types)
            region = random.choice(["upper_left", "upper_right", "lower_left", "lower_right", "central"])
            img_seed = seed + i * n_per_image + j

            try:
                result = inject_pathology(
                    pipe=pipe,
                    image_path=image_path,
                    pathology_type=pathology,
                    region=region,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=img_seed,
                    output_size=output_size,
                )

                out_filename = f"{source_id}_inject_{j}_{pathology}.png"
                out_path = os.path.join(output_dir, out_filename)
                result["injected_image"].save(out_path)

                mask_filename = f"{source_id}_inject_{j}_{pathology}_mask.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_path, result["mask"])

                record = {
                    "image_id": f"{source_id}_inject_{j}_{pathology}",
                    "image_path": out_path,
                    "source_image_id": source_id,
                    "diseases": [pathology],
                    "is_manipulated": True,
                    "manipulation_type": "injection",
                    "manipulation_mask_path": mask_path,
                    "manipulation_bbox": result["bbox"],
                    "manipulation_intensity": result["intensity"],
                    "generator": "roentgen_v1",
                    "prompt": result["prompt"],
                    "seed": img_seed,
                }
                manifest.append(record)
                success_count += 1

                if success_count % 100 == 0:
                    print(f"[Injection Pipeline] Progress: {success_count} generated...")
                    _save_manifest(manifest, manifest_path)

            except Exception as e:
                print(f"[Injection Pipeline] ERROR on {source_id} j={j}: {e}")
                fail_count += 1
                continue

    _save_manifest(manifest, manifest_path)

    print("\n[Injection Pipeline] Complete.")
    print(f"  Success: {success_count} | Failed: {fail_count}")
    print(f"  Manifest saved to: {manifest_path}")


def _save_manifest(manifest: List[Dict], manifest_path: str) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# --- QUICK SINGLE-IMAGE TEST --------------------------------

def test_single_injection(
    image_path: str,
    output_path: str = "outputs/test_injection.png",
    mask_path: str = "outputs/test_injection_mask.png",
    pathology_type: str = "nodule",
    device: str = "cuda",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pipe = load_roentgen_pipeline(device=device)

    print(f"[Test] Injecting '{pathology_type}' into: {image_path}")
    result = inject_pathology(
        pipe=pipe,
        image_path=image_path,
        pathology_type=pathology_type,
        region="random",
        seed=42,
    )

    result["injected_image"].save(output_path)
    cv2.imwrite(mask_path, result["mask"])

    print(f"[Test] Injected image saved: {output_path}")
    print(f"[Test] Mask saved:           {mask_path}")
    print(f"[Test] BBox:                 {result['bbox']}")
    print(f"[Test] Intensity:            {result['intensity']:.4f}")
    print(f"[Test] Prompt used:          {result['prompt']}")


# --- CLI ----------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoentGen Injection Attack Pipeline")
    subparsers = parser.add_subparsers(dest="mode")

    test_parser = subparsers.add_parser("test", help="Single image injection test")
    test_parser.add_argument("--image", required=True, help="Path to healthy X-ray")
    test_parser.add_argument("--pathology", default="nodule", choices=list(INJECTION_PROMPTS.keys()))
    test_parser.add_argument("--output", default="outputs/test_injection.png")
    test_parser.add_argument("--device", default="cuda")

    batch_parser = subparsers.add_parser("batch", help="Full batch injection pipeline")
    batch_parser.add_argument("--source_csv", required=True)
    batch_parser.add_argument("--output_dir", default="data/synthetic/injected")
    batch_parser.add_argument("--masks_dir", default="data/masks")
    batch_parser.add_argument("--manifest_path", default="data/synthetic/injected/manifest.json")
    batch_parser.add_argument("--n_per_image", type=int, default=1)
    batch_parser.add_argument("--max_images", type=int, default=None)
    batch_parser.add_argument("--steps", type=int, default=50)
    batch_parser.add_argument("--guidance", type=float, default=7.5)
    batch_parser.add_argument("--device", default="cuda")
    batch_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "test":
        test_single_injection(
            image_path=args.image,
            output_path=args.output,
            pathology_type=args.pathology,
            device=args.device,
        )
    elif args.mode == "batch":
        run_injection_pipeline(
            source_csv=args.source_csv,
            output_dir=args.output_dir,
            masks_dir=args.masks_dir,
            manifest_path=args.manifest_path,
            n_per_image=args.n_per_image,
            max_images=args.max_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            device=args.device,
            seed=args.seed,
        )
    else:
        parser.print_help()
