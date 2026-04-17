"""
prepare_dev_subset.py
=====================
Builds train / val / test split CSVs from the CheXpert dataset only.
Targets:
    Train : 20,000 images  (~67%)
    Val   :  3,334 images  (~11%)
    Test  : 10,000 images  (~33%)

Run from project root:
    python prepare_dev_subset.py

Outputs (written to data/splits/):
    train.csv   val.csv   test.csv
"""

import os
import sys
import random
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
CHEXPERT_ROOT  = os.path.join(PROJECT_ROOT, "data", "raw", "chexpert")
TRAIN_CSV      = os.path.join(CHEXPERT_ROOT, "train.csv")
VALID_CSV      = os.path.join(CHEXPERT_ROOT, "valid.csv")
SPLITS_DIR     = os.path.join(PROJECT_ROOT, "data", "splits")

# Confirmed CheXpert disease cols that overlap with PROJECT disease list
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

# CheXpert uses these alternative column names
CHEX_COL_MAP = {
    "Pleural Effusion": "Effusion",
    "Lung Opacity":     "Infiltration",
    "Pleural Other":    "Pleural_Thickening",
}

TARGET_TRAIN = 20_000
TARGET_VAL   =  3_334
TARGET_TEST  = 10_000
RANDOM_SEED  = 42


def _resolve_image_path(raw_path: str) -> str:
    """
    CheXpert train.csv paths look like:
        CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
    We only have the data starting from 'train/' so we strip the prefix.
    Also handles macOS '._' metadata files.
    """
    if not isinstance(raw_path, str):
        return ""

    # Normalise separators
    p = raw_path.replace("\\", "/").strip()

    # Strip known dataset prefix e.g. "CheXpert-v1.0/"
    for prefix in ("CheXpert-v1.0/", "CheXpert-v1.0-small/"):
        if p.startswith(prefix):
            p = p[len(prefix):]

    full_path = os.path.join(CHEXPERT_ROOT, p)
    return full_path


def _parse_diseases(row: pd.Series, disease_cols: list) -> str:
    """Return pipe-separated confirmed diseases (value == 1.0)."""
    diseases = []
    for col in disease_cols:
        mapped = CHEX_COL_MAP.get(col, col)
        if mapped in DISEASE_CLASSES and row.get(col, 0) == 1.0:
            diseases.append(mapped)
    return "|".join(diseases)


def load_chexpert(csv_path: str) -> list:
    """Load a CheXpert CSV and return list of record dicts."""
    if not os.path.exists(csv_path):
        print(f"  [WARN] Not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    disease_cols = [
        c for c in df.columns
        if c not in ("Path", "Sex", "Age", "Frontal/Lateral", "AP/PA")
    ]

    records = []
    for _, row in df.iterrows():
        img_path = _resolve_image_path(row.get("Path", ""))

        # Skip missing or macOS artifact files
        if not img_path or not os.path.exists(img_path):
            continue
        if os.path.basename(img_path).startswith("._"):
            continue

        diseases = _parse_diseases(row, disease_cols)
        records.append({
            "image_id":               os.path.basename(img_path),
            "image_path":             img_path,
            "diseases":               diseases,
            "is_manipulated":         False,
            "manipulation_type":      "none",
            "manipulation_mask_path": "",
            "manipulation_bbox":      "",
            "manipulation_intensity": 0.0,
            "generator":              "none",
        })

    return records


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  Dev Subset Builder — CheXpert only")
    print("=" * 60)

    # Load both CSVs
    print("\n[1/3]  Loading CheXpert train split …")
    records = load_chexpert(TRAIN_CSV)
    print(f"        Valid images found: {len(records):,}")

    print("[1/3b] Loading CheXpert valid split …")
    valid_records = load_chexpert(VALID_CSV)
    print(f"        Valid images found: {len(valid_records):,}")

    all_records = records + valid_records
    print(f"\n       Total usable images : {len(all_records):,}")

    if len(all_records) == 0:
        print("\n[ERROR] No images found.  Check that CheXpert is under:")
        print(f"        {CHEXPERT_ROOT}")
        sys.exit(1)

    # Shuffle
    random.shuffle(all_records)

    needed = TARGET_TRAIN + TARGET_VAL + TARGET_TEST
    if len(all_records) < needed:
        print(f"\n[WARN] Only {len(all_records):,} images available "
              f"but {needed:,} requested — using all available.")
        # Scale proportionally
        total = len(all_records)
        TARGET_TRAIN_USE = int(total * 0.60)
        TARGET_VAL_USE   = int(total * 0.10)
        TARGET_TEST_USE  = total - TARGET_TRAIN_USE - TARGET_VAL_USE
    else:
        TARGET_TRAIN_USE = TARGET_TRAIN
        TARGET_VAL_USE   = TARGET_VAL
        TARGET_TEST_USE  = TARGET_TEST

    train_records = all_records[:TARGET_TRAIN_USE]
    val_records   = all_records[TARGET_TRAIN_USE : TARGET_TRAIN_USE + TARGET_VAL_USE]
    test_records  = all_records[TARGET_TRAIN_USE + TARGET_VAL_USE :
                                TARGET_TRAIN_USE + TARGET_VAL_USE + TARGET_TEST_USE]

    # Write splits
    print("\n[2/3]  Writing split CSVs …")
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path   = os.path.join(SPLITS_DIR, "val.csv")
    test_path  = os.path.join(SPLITS_DIR, "test.csv")

    pd.DataFrame(train_records).to_csv(train_path, index=False)
    pd.DataFrame(val_records).to_csv(val_path,   index=False)
    pd.DataFrame(test_records).to_csv(test_path,  index=False)

    print(f"\n[3/3]  Done!\n")
    print(f"  Train : {len(train_records):>7,} images  →  {train_path}")
    print(f"  Val   : {len(val_records):>7,} images  →  {val_path}")
    print(f"  Test  : {len(test_records):>7,} images  →  {test_path}")
    print()

    # Quick sanity: peek at first row
    first = train_records[0]
    print("  Sample row:")
    print(f"    image_id   : {first['image_id']}")
    print(f"    image_path : {first['image_path']}")
    print(f"    diseases   : '{first['diseases']}'")
    print(f"    is_manip   : {first['is_manipulated']}")
    print()
    print("  All splits ready. Run training next:")
    print("    python -m src.training.run_phase1")


if __name__ == "__main__":
    main()
