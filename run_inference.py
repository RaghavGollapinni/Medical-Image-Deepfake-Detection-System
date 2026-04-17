"""
run_inference.py
================
CLI tool for single-image and batch inference (PRD D1.4).

Usage:
    # Single image
    python run_inference.py --image path/to/xray.jpg

    # Batch (directory of images)
    python run_inference.py --dir path/to/images/

    # With heatmap overlays saved
    python run_inference.py --image path/to/xray.jpg --heatmap

Output:
    Prints Verified Diagnosis JSON to stdout.
    Saves heatmap PNGs to outputs/heatmaps/ if --heatmap is specified.
"""

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import load_config
from src.inference.predict import DeepfakePredictor

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Medical Deepfake Detector - Inference CLI"
    )
    parser.add_argument("--image", type=str, help="Path to a single X-ray image")
    parser.add_argument("--dir", type=str, help="Path to directory of X-ray images")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (default: best_forgery_auc.pt)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate and save heatmap overlays")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: outputs/heatmaps)")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="Config YAML path")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide --image or --dir")

    # Load config
    cfg = load_config(args.config)

    # Determine checkpoint
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt_dir = cfg["paths"].get("checkpoints", "checkpoints")
        ckpt = os.path.join(ckpt_dir, "best_forgery_auc.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ckpt_dir, "best_model.pt")

    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("  Train the model first: python -m src.training.run_phase1")
        sys.exit(1)

    # Output directory
    out_dir = args.output_dir or cfg["paths"].get("heatmaps", "outputs/heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # Build predictor
    print("Loading model ...")
    predictor = DeepfakePredictor(cfg, ckpt)
    print("Ready.\n")

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(glob.glob(os.path.join(args.dir, ext)))

    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    # Run inference
    all_results = []
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)} ... ", end="")

        try:
            result, overlay = predictor.predict(img_path, generate_heatmap=args.heatmap)

            if args.heatmap and overlay is not None:
                heatmap_path = os.path.join(
                    out_dir,
                    f"{os.path.splitext(os.path.basename(img_path))[0]}_heatmap.png"
                )
                cv2.imwrite(heatmap_path, overlay)

            trust_label = result["recommendation"]
            print(f"Trust={result['trust_score']:.2f} | {trust_label}")
            all_results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"error": str(e), "image": img_path})

    # Output JSON
    if len(all_results) == 1:
        print("\n" + json.dumps(all_results[0], indent=2))
    else:
        results_path = os.path.join(out_dir, "batch_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBatch results saved: {results_path}")


if __name__ == "__main__":
    main()
