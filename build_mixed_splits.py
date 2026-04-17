import os
import json
import pandas as pd
import random

def main():
    splits_dir = r"D:\Projects and Research\VAC - Healthcare Security\medical_deepfake_detector\data\splits"
    manifest_path = r"D:\Projects and Research\VAC - Healthcare Security\medical_deepfake_detector\data\synthetic\subtle\freq_manifest.json"
    
    if not os.path.exists(manifest_path):
        print("Manifest not found:", manifest_path)
        return
        
    print("Loading synthetic manifest...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    random.seed(42)
    random.shuffle(manifest)
    
    # Let's say manifest has 10,000 items. Let's put 7000 to train, 1000 to val, 2000 to test
    # Or just use the proportions: 70%, 10%, 20%
    n_total = len(manifest)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)
    
    fake_train = manifest[:n_train]
    fake_val = manifest[n_train:n_train+n_val]
    fake_test = manifest[n_train+n_val:]
    
    print(f"Generated fake splits: Train: {len(fake_train)}, Val: {len(fake_val)}, Test: {len(fake_test)}")
    
    # Load authentic splits
    # Assuming original file has only real images. We'll load the original files if they exist, or the ones we have.
    # To be safe, we always use the unmixed files. 
    # If train_real.csv exists, load it. Otherwise, assume train.csv is real and rename it.
    for split_name, fake_data in zip(["train", "val", "test"], [fake_train, fake_val, fake_test]):
        real_csv = os.path.join(splits_dir, f"{split_name}_real.csv")
        curr_csv = os.path.join(splits_dir, f"{split_name}.csv")
        
        # Backup original (authentic-only) CSV as real
        if not os.path.exists(real_csv):
            if os.path.exists(curr_csv):
                os.rename(curr_csv, real_csv)
            else:
                print(f"File {curr_csv} not found.")
                continue
                
        df_real = pd.read_csv(real_csv)
        df_fake = pd.DataFrame(fake_data)
        
        # Combine
        df_mixed = pd.concat([df_real, df_fake], ignore_index=True)
        # Shuffle
        df_mixed = df_mixed.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to the main CSV
        df_mixed.to_csv(curr_csv, index=False)
        print(f"[{split_name.upper()}] Mixed generated: {len(df_mixed)} items (Real: {len(df_real)}, Fake: {len(df_fake)})")

if __name__ == "__main__":
    main()
