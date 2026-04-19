import os
import cv2
import numpy as np
import pandas as pd

def bootstrap():
    # 1. Create directory for raw images
    os.makedirs("data/raw/bootstrap/images", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    records = []
    print("Generating 100 bootstrap images for Colab pipeline test...")

    for i in range(100):
        # Create a dummy X-ray like image (Grayscale noise + circle)
        img = np.zeros((224, 224), dtype=np.uint8)
        # Draw a lung-like oval
        cv2.ellipse(img, (112, 112), (60, 90), 0, 0, 360, (60), -1)
        # Add noise
        noise = np.random.normal(0, 15, (224, 224)).astype(np.uint8)
        img = cv2.add(img, noise)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        img_name = f"boot_{i}.png"
        img_path = os.path.join("data/raw/bootstrap/images", img_name)
        cv2.imwrite(img_path, img)
        
        # Add alternating diseases
        diseases = ""
        if i % 10 == 0: diseases = "Pneumonia"
        elif i % 15 == 0: diseases = "Effusion"
        
        records.append({
            "image_id": img_name,
            "image_path": img_path,
            "diseases": diseases,
            "is_manipulated": False,
            "manipulation_type": "none",
            "manipulation_mask_path": "",
            "manipulation_bbox": "",
            "manipulation_intensity": 0.0,
            "generator": "none"
        })

    # 2. Save to splits
    df = pd.DataFrame(records)
    df.to_csv("data/splits/train.csv", index=False)
    
    # Val and Test (small subsets)
    df.head(20).to_csv("data/splits/val.csv", index=False)
    df.head(20).to_csv("data/splits/test.csv", index=False)

    print(f"✅ Success: Successfully bootstrapped 100 images to data/splits/train.csv")
    print(f"Next step: Run '!python -m deepfake_pipeline.run_pipeline --max_images 50'")

if __name__ == "__main__":
    bootstrap()
