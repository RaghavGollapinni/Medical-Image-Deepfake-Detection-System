# ============================================================
# deepfake_pipeline/cyclegan_transform.py
# Forensic-Aware Medical Deepfake Detection System
# VAC - Healthcare Security Project | PRD v2.0
#
# SUBTLE MANIPULATION via CycleGAN Domain Transformation
# -------------------------------------------------------
# Performs unpaired image-to-image translation between
# healthy and diseased chest X-ray domains using CycleGAN.
# ============================================================

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# --- CYCLEGAN GENERATOR ARCHITECTURE ------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, num_residual_blocks: int = 9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        in_ch = 64
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(in_ch))

        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, in_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- WEIGHT LOADING -----------------------------------------

def load_generator(
    weights_path: str,
    device: str = "cuda",
    num_residual_blocks: int = 9,
) -> CycleGANGenerator:
    model = CycleGANGenerator(num_residual_blocks=num_residual_blocks)
    state_dict = torch.load(weights_path, map_location=device)

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"[CycleGAN] Generator loaded from: {weights_path}")
    return model


def load_generators_pair(
    g_healthy2disease_path: str,
    g_disease2healthy_path: str,
    device: str = "cuda",
) -> Tuple[CycleGANGenerator, CycleGANGenerator]:
    G = load_generator(g_healthy2disease_path, device)
    F = load_generator(g_disease2healthy_path, device)
    return G, F


# --- IMAGE TRANSFORMS ---------------------------------------

def get_inference_transform(image_size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


# --- INTENSITY BLENDING -------------------------------------

def blend_with_source(
    source_pil: Image.Image,
    transformed_pil: Image.Image,
    intensity: float,
) -> Image.Image:
    src_np = np.array(source_pil).astype(np.float32)
    tgt_np = np.array(transformed_pil).astype(np.float32)
    blend_np = (1.0 - intensity) * src_np + intensity * tgt_np
    return Image.fromarray(blend_np.clip(0, 255).astype(np.uint8))


# --- CORE TRANSFORM FUNCTION --------------------------------

def transform_image(
    generator: CycleGANGenerator,
    image_path: str,
    direction: str,
    intensity: float = 1.0,
    output_size: int = 224,
    device: str = "cuda",
) -> Dict:
    source_pil = Image.open(image_path).convert("RGB")
    transform = get_inference_transform(image_size=256)
    input_tensor = transform(source_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    transformed_pil = tensor_to_pil(output_tensor)

    source_resized = source_pil.resize((output_size, output_size), Image.LANCZOS)
    transformed_resized = transformed_pil.resize((output_size, output_size), Image.LANCZOS)
    final_image = blend_with_source(source_resized, transformed_resized, intensity)

    metadata = {
        "source_image": image_path,
        "direction": direction,
        "intensity": round(intensity, 4),
        "generator": "cyclegan_v1",
        "manipulation_type": "subtle",
        "timestamp": datetime.utcnow().isoformat(),
    }

    return {
        "transformed_image": final_image,
        "intensity": intensity,
        "direction": direction,
        "metadata": metadata,
    }


# --- BATCH TRANSFORM PIPELINE -------------------------------

def run_transform_pipeline(
    source_csv: str,
    output_dir: str,
    manifest_path: str,
    g_healthy2disease_path: str,
    g_disease2healthy_path: str,
    direction: str = "both",
    intensity_range: Tuple[float, float] = (0.2, 0.8),
    subtle_fraction: float = 0.5,
    output_size: int = 224,
    device: str = "cuda",
    max_images: Optional[int] = None,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    df = pd.read_csv(source_csv)
    if max_images:
        df = df.head(max_images)

    G, F = load_generators_pair(g_healthy2disease_path, g_disease2healthy_path, device=device)

    healthy_mask = df["diseases"].isna() | (df["diseases"] == "") | (df["diseases"].str.lower() == "no finding")
    healthy_df = df[healthy_mask].reset_index(drop=True)
    diseased_df = df[~healthy_mask].reset_index(drop=True)

    print(f"[CycleGAN] Healthy images:  {len(healthy_df)}")
    print(f"[CycleGAN] Diseased images: {len(diseased_df)}")

    manifest = []
    success_count = 0
    fail_count = 0

    def process_batch(source_df, generator, gen_direction):
        nonlocal success_count, fail_count

        for _, row in source_df.iterrows():
            image_path = row["image_path"]
            source_id = str(row["image_id"])

            if not os.path.exists(image_path):
                fail_count += 1
                continue

            if random.random() < subtle_fraction:
                intensity = random.uniform(0.05, 0.30)
            else:
                intensity = random.uniform(*intensity_range)

            try:
                result = transform_image(
                    generator=generator,
                    image_path=image_path,
                    direction=gen_direction,
                    intensity=intensity,
                    output_size=output_size,
                    device=device,
                )

                out_filename = f"{source_id}_cyclegan_{gen_direction}.png"
                out_path = os.path.join(output_dir, out_filename)
                result["transformed_image"].save(out_path)

                record = {
                    "image_id": f"{source_id}_cyclegan_{gen_direction}",
                    "image_path": out_path,
                    "source_image_id": source_id,
                    "diseases": str(row.get("diseases", "")),
                    "is_manipulated": True,
                    "manipulation_type": "subtle",
                    "manipulation_mask_path": "",
                    "manipulation_bbox": "",
                    "manipulation_intensity": round(intensity, 4),
                    "generator": "cyclegan_v1",
                    "direction": gen_direction,
                }
                manifest.append(record)
                success_count += 1

                if success_count % 100 == 0:
                    print(f"[CycleGAN] Progress: {success_count} generated...")
                    _save_manifest(manifest, manifest_path)

            except Exception as e:
                print(f"[CycleGAN] ERROR on {source_id}: {e}")
                fail_count += 1

    if direction in ["healthy2disease", "both"]:
        print(f"[CycleGAN] Running healthy -> diseased on {len(healthy_df)} images...")
        process_batch(healthy_df, G, "healthy2disease")

    if direction in ["disease2healthy", "both"]:
        print(f"[CycleGAN] Running diseased -> healthy on {len(diseased_df)} images...")
        process_batch(diseased_df, F, "disease2healthy")

    _save_manifest(manifest, manifest_path)

    print("\n[CycleGAN] Complete.")
    print(f"  Success: {success_count} | Failed: {fail_count}")
    print(f"  Manifest saved to: {manifest_path}")


def _save_manifest(manifest: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# --- TRAINING UTILITIES -------------------------------------

def get_training_notes() -> str:
    return """
============================================================
CycleGAN TRAINING NOTES FOR CHEST X-RAYS
============================================================
If pretrained weights are not available, train CycleGAN
from scratch using NIH ChestX-ray14:
  Domain A (Healthy):  Images with "No Finding" label
  Domain B (Diseased): Images with any pathology label
============================================================
"""


# --- QUICK TEST ---------------------------------------------

def test_single_transform(
    image_path: str,
    generator_path: str,
    direction: str = "healthy2disease",
    intensity: float = 0.7,
    output_path: str = "outputs/test_cyclegan.png",
    device: str = "cuda",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = load_generator(generator_path, device=device)

    print(f"[Test] Transforming: {image_path}")
    print(f"[Test] Direction:    {direction}")
    print(f"[Test] Intensity:    {intensity}")

    result = transform_image(
        generator=generator,
        image_path=image_path,
        direction=direction,
        intensity=intensity,
        device=device,
    )

    result["transformed_image"].save(output_path)
    print(f"[Test] Saved: {output_path}")


# --- CLI ----------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN Transformation Pipeline for Subtle Attacks")
    subparsers = parser.add_subparsers(dest="mode")

    tp = subparsers.add_parser("test")
    tp.add_argument("--image", required=True)
    tp.add_argument("--weights", required=True, help="Generator .pth weights")
    tp.add_argument("--direction", default="healthy2disease", choices=["healthy2disease", "disease2healthy"])
    tp.add_argument("--intensity", type=float, default=0.7)
    tp.add_argument("--output", default="outputs/test_cyclegan.png")
    tp.add_argument("--device", default="cuda")

    bp = subparsers.add_parser("batch")
    bp.add_argument("--source_csv", required=True)
    bp.add_argument("--output_dir", default="data/synthetic/subtle")
    bp.add_argument("--manifest_path", default="data/synthetic/subtle/manifest.json")
    bp.add_argument("--g_h2d", required=True, help="healthy->disease generator weights")
    bp.add_argument("--g_d2h", required=True, help="disease->healthy generator weights")
    bp.add_argument("--direction", default="both", choices=["healthy2disease", "disease2healthy", "both"])
    bp.add_argument("--subtle_fraction", type=float, default=0.5)
    bp.add_argument("--max_images", type=int, default=None)
    bp.add_argument("--device", default="cuda")
    bp.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("train_info")

    args = parser.parse_args()

    if args.mode == "test":
        test_single_transform(
            image_path=args.image,
            generator_path=args.weights,
            direction=args.direction,
            intensity=args.intensity,
            output_path=args.output,
            device=args.device,
        )
    elif args.mode == "batch":
        run_transform_pipeline(
            source_csv=args.source_csv,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            g_healthy2disease_path=args.g_h2d,
            g_disease2healthy_path=args.g_d2h,
            direction=args.direction,
            subtle_fraction=args.subtle_fraction,
            max_images=args.max_images,
            device=args.device,
            seed=args.seed,
        )
    elif args.mode == "train_info":
        print(get_training_notes())
    else:
        parser.print_help()
