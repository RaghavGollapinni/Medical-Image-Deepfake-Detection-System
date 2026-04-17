# ============================================================
# deepfake_pipeline/roentgen_erase.py
# Forensic-Aware Medical Deepfake Detection System
# VAC - Healthcare Security Project | PRD v2.0
#
# ERASURE ATTACKS via RoentGen Inpainting
# ----------------------------------------
# Takes a DISEASED chest X-ray and erases the existing pathology
# by inpainting the masked region with healthy-looking tissue.
# Uses RoentGen - a domain-adapted Stable Diffusion model
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


# --- ERASURE PROMPTS ----------------------------------------

ERASURE_PROMPTS = {
    "general": [
        "normal chest x-ray, clear lung fields, no focal opacity",
        "frontal chest radiograph showing normal lung parenchyma, no infiltrates",
        "chest x-ray with clear lungs bilaterally, no consolidation or effusion",
        "normal posteroanterior chest radiograph, healthy lung tissue",
    ],
    "upper_lobe": [
        "normal chest x-ray showing clear upper lobe lung fields",
        "frontal chest radiograph with normal upper lobe parenchyma, no apical opacity",
    ],
    "lower_lobe": [
        "normal chest x-ray with clear costophrenic angles and lower lobe fields",
        "frontal chest radiograph showing normal lower lobe lung parenchyma",
    ],
    "perihilar": [
        "normal chest x-ray with clear perihilar regions, no hilar adenopathy",
        "frontal chest radiograph showing normal hilar structures",
    ],
}

NEGATIVE_PROMPT = (
    "consolidation, opacity, infiltrate, effusion, nodule, mass, lesion, "
    "pneumonia, atelectasis, blurry, artifacts, watermark, low quality"
)


# --- MASK BUILDERS ------------------------------------------

def build_mask_from_bbox(
    bbox: List[int],
    image_size: int,
    dilation_px: int = 12,
) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    x, y, w, h = bbox

    x1 = max(0, x - dilation_px)
    y1 = max(0, y - dilation_px)
    x2 = min(image_size, x + w + dilation_px)
    y2 = min(image_size, y + h + dilation_px)

    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    return mask


def build_mask_from_segmentation(
    seg_mask_path: str,
    image_size: int,
    dilation_px: int = 8,
) -> np.ndarray:
    mask_raw = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Mask not found: {seg_mask_path}")

    mask_raw = cv2.resize(mask_raw, (image_size, image_size))
    _, mask_binary = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
    return mask_binary


def estimate_region_from_bbox(bbox: List[int], image_size: int) -> str:
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    third = image_size // 3

    if cy < third:
        return "upper_lobe"
    if cy > 2 * third:
        return "lower_lobe"
    if abs(cx - image_size // 2) < image_size // 5:
        return "perihilar"
    return "general"


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


# --- CORE ERASURE FUNCTION ----------------------------------

def erase_pathology(
    pipe: StableDiffusionInpaintPipeline,
    image_path: str,
    bbox: Optional[List[int]] = None,
    seg_mask_path: Optional[str] = None,
    disease_label: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    output_size: int = 224,
    dilation_px: int = 12,
) -> Dict:
    if bbox is None and seg_mask_path is None:
        raise ValueError("Must provide either bbox or seg_mask_path to define the erasure region.")

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
        seed = random.randint(0, 999999)

    source_img = Image.open(image_path).convert("RGB")
    source_img = source_img.resize((output_size, output_size), Image.LANCZOS)

    if seg_mask_path is not None and os.path.exists(seg_mask_path):
        mask_np = build_mask_from_segmentation(seg_mask_path, output_size, dilation_px)
        final_bbox = _bbox_from_mask(mask_np)
    else:
        original_w, original_h = source_img.size
        scale_x = output_size / original_w
        scale_y = output_size / original_h
        scaled_bbox = [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ]
        mask_np = build_mask_from_bbox(scaled_bbox, output_size, dilation_px)
        final_bbox = scaled_bbox

    mask_pil = Image.fromarray(mask_np).convert("RGB")
    region = estimate_region_from_bbox(final_bbox, output_size)
    prompt = random.choice(ERASURE_PROMPTS.get(region, ERASURE_PROMPTS["general"]))

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=source_img,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    erased_image = result.images[0]

    mask_coverage = (mask_np > 127).sum() / (output_size * output_size)
    intensity = float(np.clip(mask_coverage * 5.0, 0.1, 1.0))

    metadata = {
        "source_image": image_path,
        "original_disease": disease_label or "unknown",
        "region": region,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "bbox": final_bbox,
        "dilation_px": dilation_px,
        "intensity": round(intensity, 4),
        "generator": "roentgen_v1",
        "manipulation_type": "erasure",
        "timestamp": datetime.utcnow().isoformat(),
    }

    return {
        "erased_image": erased_image,
        "mask": mask_np,
        "bbox": final_bbox,
        "prompt": prompt,
        "intensity": intensity,
        "metadata": metadata,
    }


def _bbox_from_mask(mask_np: np.ndarray) -> List[int]:
    coords = cv2.findNonZero(mask_np)
    if coords is None:
        h, w = mask_np.shape
        return [0, 0, w, h]
    x, y, w, h = cv2.boundingRect(coords)
    return [x, y, w, h]


# --- BATCH ERASURE PIPELINE ---------------------------------

def run_erasure_pipeline(
    source_csv: str,
    output_dir: str,
    masks_dir: str,
    manifest_path: str,
    bbox_csv: Optional[str] = None,
    device: str = "cuda",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    output_size: int = 224,
    max_images: Optional[int] = None,
    seed: int = 42,
    dilation_px: int = 12,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    df = pd.read_csv(source_csv)
    diseased_df = df[
        df["diseases"].notna()
        & (df["diseases"] != "")
        & (df["diseases"].str.lower() != "no finding")
    ].reset_index(drop=True)

    if max_images is not None:
        diseased_df = diseased_df.head(max_images)

    bbox_lookup = {}
    if bbox_csv is not None and os.path.exists(bbox_csv) and os.path.getsize(bbox_csv) > 0:
        bbox_df = pd.read_csv(bbox_csv)
        for _, row in bbox_df.iterrows():
            img_id = row["Image Index"]
            if img_id not in bbox_lookup:
                bbox_lookup[img_id] = []
            bbox_lookup[img_id].append([
                int(row["Bbox [x"]),
                int(row["y"]),
                int(row["w"]),
                int(row["h]"]),
            ])

    print(f"[Erasure Pipeline] Source diseased images: {len(diseased_df)}")
    print(f"[Erasure Pipeline] GT bboxes loaded:       {len(bbox_lookup)}")

    pipe = load_roentgen_pipeline(device=device)

    manifest = []
    success_count = 0
    fail_count = 0

    for i, row in diseased_df.iterrows():
        image_path = row["image_path"]
        source_id = str(row["image_id"])

        if not os.path.exists(image_path):
            print(f"[Erasure Pipeline] SKIP (not found): {image_path}")
            fail_count += 1
            continue

        seg_mask_path = str(row.get("manipulation_mask_path", ""))
        seg_mask_path = seg_mask_path if os.path.exists(seg_mask_path) else None

        bbox = None
        if seg_mask_path is None:
            if source_id in bbox_lookup and bbox_lookup[source_id]:
                bbox = random.choice(bbox_lookup[source_id])
            else:
                center = output_size // 4
                size = output_size // 3
                bbox = [center, center, size, size]

        diseases_str = str(row.get("diseases", ""))
        disease_label = diseases_str.split("|")[0] if diseases_str else "unknown"
        img_seed = seed + i

        try:
            result = erase_pathology(
                pipe=pipe,
                image_path=image_path,
                bbox=bbox,
                seg_mask_path=seg_mask_path,
                disease_label=disease_label,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=img_seed,
                output_size=output_size,
                dilation_px=dilation_px,
            )

            out_filename = f"{source_id}_erased.png"
            out_path = os.path.join(output_dir, out_filename)
            result["erased_image"].save(out_path)

            mask_filename = f"{source_id}_erased_mask.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, result["mask"])

            record = {
                "image_id": f"{source_id}_erased",
                "image_path": out_path,
                "source_image_id": source_id,
                "original_disease": disease_label,
                "diseases": "",
                "is_manipulated": True,
                "manipulation_type": "erasure",
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
                print(f"[Erasure Pipeline] Progress: {success_count} generated...")
                _save_manifest(manifest, manifest_path)

        except Exception as e:
            print(f"[Erasure Pipeline] ERROR on {source_id}: {e}")
            fail_count += 1
            continue

    _save_manifest(manifest, manifest_path)

    print("\n[Erasure Pipeline] Complete.")
    print(f"  Success: {success_count} | Failed: {fail_count}")
    print(f"  Manifest saved to: {manifest_path}")


def _save_manifest(manifest: List[Dict], manifest_path: str) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# --- QUICK SINGLE-IMAGE TEST --------------------------------

def test_single_erasure(
    image_path: str,
    bbox: List[int],
    output_path: str = "outputs/test_erasure.png",
    mask_path: str = "outputs/test_erasure_mask.png",
    device: str = "cuda",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pipe = load_roentgen_pipeline(device=device)

    print(f"[Test] Erasing pathology from: {image_path}")
    print(f"[Test] BBox: {bbox}")

    result = erase_pathology(
        pipe=pipe,
        image_path=image_path,
        bbox=bbox,
        seed=42,
    )

    result["erased_image"].save(output_path)
    cv2.imwrite(mask_path, result["mask"])

    print(f"[Test] Erased image saved: {output_path}")
    print(f"[Test] Mask saved:         {mask_path}")
    print(f"[Test] Intensity:          {result['intensity']:.4f}")
    print(f"[Test] Prompt used:        {result['prompt']}")


# --- CLI ----------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoentGen Erasure Attack Pipeline")
    subparsers = parser.add_subparsers(dest="mode")

    test_parser = subparsers.add_parser("test", help="Single image erasure test")
    test_parser.add_argument("--image", required=True, help="Path to diseased X-ray")
    test_parser.add_argument("--bbox", required=True, nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    test_parser.add_argument("--output", default="outputs/test_erasure.png")
    test_parser.add_argument("--device", default="cuda")

    batch_parser = subparsers.add_parser("batch", help="Full batch erasure pipeline")
    batch_parser.add_argument("--source_csv", required=True)
    batch_parser.add_argument("--output_dir", default="data/synthetic/erased")
    batch_parser.add_argument("--masks_dir", default="data/masks")
    batch_parser.add_argument("--manifest_path", default="data/synthetic/erased/manifest.json")
    batch_parser.add_argument("--bbox_csv", default=None, help="NIH BBox_List_2017.csv for ground-truth bboxes")
    batch_parser.add_argument("--max_images", type=int, default=None)
    batch_parser.add_argument("--steps", type=int, default=50)
    batch_parser.add_argument("--guidance", type=float, default=7.5)
    batch_parser.add_argument("--dilation", type=int, default=12)
    batch_parser.add_argument("--device", default="cuda")
    batch_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "test":
        test_single_erasure(
            image_path=args.image,
            bbox=args.bbox,
            output_path=args.output,
            device=args.device,
        )
    elif args.mode == "batch":
        run_erasure_pipeline(
            source_csv=args.source_csv,
            output_dir=args.output_dir,
            masks_dir=args.masks_dir,
            manifest_path=args.manifest_path,
            bbox_csv=args.bbox_csv,
            max_images=args.max_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            dilation_px=args.dilation,
            device=args.device,
            seed=args.seed,
        )
    else:
        parser.print_help()
