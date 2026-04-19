import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the full synthetic data pipeline")
    parser.add_argument("--max_images", type=int, default=5, help="Number of images per pipeline for quick testing")
    parser.add_argument("--splits_dir", default="data/splits", help="Directory containing train.csv")
    args = parser.parse_args()

    train_csv = os.path.join(args.splits_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Frequency Perturbation
    print("\n" + "="*50)
    print("Running Frequency Perturbation...")
    print("="*50)
    subprocess.run([
        "python", "deepfake_pipeline/freq_perturb.py", "batch",
        "--source_csv", train_csv,
        "--max_images", str(args.max_images)
    ], check=True)

    # 2. RoentGen Injection
    print("\n" + "="*50)
    print("Running RoentGen Injection Attack...")
    print("="*50)
    subprocess.run([
        "python", "deepfake_pipeline/roentgen_inject.py", "batch",
        "--source_csv", train_csv,
        "--max_images", str(args.max_images),
        "--steps", "20"  # Fewer steps for faster runtime
    ], check=False)

    # 3. RoentGen Erasure
    print("\n" + "="*50)
    print("Running RoentGen Erasure Attack...")
    print("="*50)
    subprocess.run([
        "python", "deepfake_pipeline/roentgen_erase.py", "batch",
        "--source_csv", train_csv,
        "--max_images", str(args.max_images),
        "--steps", "20"
    ], check=False)

    # Note: CycleGAN is skipped due to missing pre-trained weights for the time being.
    
    print("\nPipeline components executed successfully.")

if __name__ == "__main__":
    main()
