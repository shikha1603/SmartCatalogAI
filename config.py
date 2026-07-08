import os
import torch

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.pt")
CLASS_INDICES_PATH = os.path.join(SAVED_MODEL_DIR, "class_indices.json")
CLASS_THRESHOLDS_PATH = os.path.join(SAVED_MODEL_DIR, "class_thresholds.json")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
THUMBNAILS_DIR = os.path.join(OUTPUTS_DIR, "thumbnails")
ACCURACY_CURVE_PATH = os.path.join(OUTPUTS_DIR, "accuracy_curve.png")
LOSS_CURVE_PATH = os.path.join(OUTPUTS_DIR, "loss_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.csv")

DB_PATH = os.path.join(BASE_DIR, "database.db")

# Ensure necessary directories exist
for folder in [SAVED_MODEL_DIR, OUTPUTS_DIR, THUMBNAILS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Image size for MobileNetV2
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Categories and confidence routing threshold
CATEGORIES = ["Fashion", "Electronics", "Home", "Beauty", "Grocery"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MARGIN_THRESHOLD = 0.15

# PyTorch device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyper-parameters
EPOCHS_STAGE1 = 5      # Training dense head
EPOCHS_STAGE2 = 10     # Fine-tuning last N layers
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5

# Dataset Mapping for subfolder category identification
CATEGORY_MAP = {
    # Fashion
    "tshirt": "Fashion",
    "jeans": "Fashion",
    "apparel": "Fashion",
    "footwear": "Fashion",
    "shoes": "Fashion",
    "clothing": "Fashion",
    "boys": "Fashion",
    "girls": "Fashion",
    "men": "Fashion",
    "women": "Fashion",
    # Electronics
    "tv": "Electronics",
    "television": "Electronics",
    "mobile": "Electronics",
    "laptop": "Electronics",
    "camera": "Electronics",
    "headphones": "Electronics",
    # Home
    "sofa": "Home",
    "bed": "Home",
    "chair": "Home",
    "table": "Home",
    "lamp": "Home",
    "furniture": "Home",
    # Beauty
    "perfume": "Beauty",
    "lipstick": "Beauty",
    "makeup": "Beauty",
    "cosmetics": "Beauty",
    "lotion": "Beauty",
    # Grocery
    "grocery": "Grocery",
    "fruits": "Grocery",
    "vegetables": "Grocery",
    "snacks": "Grocery",
    "beverage": "Grocery"
}
