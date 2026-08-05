"""
Project configuration.

Defines all important directories used throughout the project.

Author: Shahriar Ahmed
"""

from pathlib import Path

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(r"D:\Pancreatic_Cancer_Thesis")


# ============================================================
# Data Directories
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_CT_DIR = DATA_DIR / "raw_ct"

LABELS_DIR = DATA_DIR / "labels"

MANUAL_LABEL_DIR = LABELS_DIR / "Manual_Labels"

AUTO_LABEL_DIR = LABELS_DIR / "Automatic_Labels"

METADATA_DIR = DATA_DIR / "metadata"

CROPPED_DIR = DATA_DIR / "cropped"

PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================
# Processed Dataset
# ============================================================

PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"

PROCESSED_MASKS_DIR = PROCESSED_DIR / "masks"

PROCESSED_METADATA_FILE = PROCESSED_DIR / "metadata.csv"

TRAIN_SPLIT_FILE = PROCESSED_DIR / "train.csv"

VAL_SPLIT_FILE = PROCESSED_DIR / "val.csv"

TEST_SPLIT_FILE = PROCESSED_DIR / "test.csv"


# ============================================================
# Project Outputs
# ============================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = PROJECT_ROOT / "figures"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ============================================================
# Create directories automatically
# ============================================================

_REQUIRED_DIRS = [
    OUTPUTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CROPPED_DIR,
    PROCESSED_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
]

for directory in _REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)