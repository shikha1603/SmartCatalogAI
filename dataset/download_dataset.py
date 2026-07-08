"""
download_dataset.py

Dataset organizer and downloader for SmartCatalog AI.
Features:
1. --path: Organize a manually downloaded ZIP file or extracted folder of product images.
2. --synthetic: Generate a high-quality synthetic dataset for testing and CI (NOT for production training).
3. Auto-download from Kaggle using Kaggle API credentials if available.

Handles class mapping, class imbalance reporting, and split organization.
"""

import os
import argparse
import shutil
import random
import zipfile
import logging
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple
from config import DATASET_DIR, TRAIN_DIR, VAL_DIR, CATEGORIES, CATEGORY_MAP

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_dataset")

def create_synthetic_image(category: str, output_path: str) -> None:
    """Generates a unique randomized synthetic 224x224 image representing a specific category."""
    # 1. Jitter background color slightly to create diverse lighting conditions
    bg_r = random.randint(235, 255)
    bg_g = random.randint(235, 255)
    bg_b = random.randint(235, 255)
    img = Image.new("RGB", (224, 224), color=(bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    
    # 2. Add randomized background noise (lines and dots) representing busy scenes
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, 224)
        y1 = random.randint(0, 224)
        x2 = random.randint(0, 224)
        y2 = random.randint(0, 224)
        line_color = (random.randint(200, 234), random.randint(200, 234), random.randint(200, 234))
        draw.line([x1, y1, x2, y2], fill=line_color, width=random.randint(1, 2))
        
    for _ in range(random.randint(10, 30)):
        x = random.randint(0, 224)
        y = random.randint(0, 224)
        dot_color = (random.randint(210, 240), random.randint(210, 240), random.randint(210, 240))
        draw.point([x, y], fill=dot_color)

    # 3. Choose category-specific randomized drawing parameters
    scale = random.uniform(0.75, 1.25)
    offset_x = random.randint(-15, 15)
    offset_y = random.randint(-15, 15)
    
    center_x = 112 + offset_x
    center_y = 112 + offset_y

    if category == "Fashion":
        # Draw randomized clothing/t-shirt silhouette
        body_color = (random.randint(150, 245), random.randint(70, 130), random.randint(50, 110))
        w = int(45 * scale)
        h = int(50 * scale)
        # Body
        draw.rectangle([center_x - w, center_y - h, center_x + w, center_y + h], fill=body_color)
        # Sleeves
        sleeve_w = int(70 * scale)
        sleeve_h = int(18 * scale)
        draw.rectangle([center_x - sleeve_w, center_y - h, center_x + sleeve_w, center_y - h + sleeve_h], fill=body_color)
        # Neck cutout (matching background color)
        neck_r = int(18 * scale)
        draw.ellipse([center_x - neck_r, center_y - h - neck_r, center_x + neck_r, center_y - h + neck_r], fill=(bg_r, bg_g, bg_b))
        
    elif category == "Electronics":
        # Draw a randomized TV screen / tablet monitor
        border_color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
        screen_color = (random.randint(0, 40), random.randint(100, 190), random.randint(150, 220))
        w = int(70 * scale)
        h = int(45 * scale)
        # Bezel/frame
        draw.rectangle([center_x - w, center_y - h, center_x + w, center_y + h], fill=border_color)
        # Screen panel inside bezel
        inset = random.randint(4, 7)
        draw.rectangle([center_x - w + inset, center_y - h + inset, center_x + w - inset, center_y + h - inset], fill=screen_color)
        # Stand neck & base
        stand_w = int(8 * scale)
        stand_h = int(25 * scale)
        draw.rectangle([center_x - stand_w, center_y + h, center_x + stand_w, center_y + h + stand_h], fill=border_color)
        base_w = int(35 * scale)
        draw.rectangle([center_x - base_w, center_y + h + stand_h, center_x + base_w, center_y + h + stand_h + int(8 * scale)], fill=border_color)
        
    elif category == "Home":
        # Draw a randomized floor lamp / desk light
        shade_color = (random.randint(220, 255), random.randint(170, 230), random.randint(50, 120))
        stand_color = (random.randint(80, 130), random.randint(80, 130), random.randint(100, 140))
        w = int(35 * scale)
        h = int(25 * scale)
        # Lampshade
        draw.polygon([
            (center_x - w, center_y + h), 
            (center_x + w, center_y + h), 
            (center_x + int(w * 0.6), center_y - h), 
            (center_x - int(w * 0.6), center_y - h)
        ], fill=shade_color)
        # Lamp stand line
        stand_bottom_y = center_y + h + int(70 * scale)
        draw.line([center_x, center_y + h, center_x, stand_bottom_y], fill=stand_color, width=max(3, int(5 * scale)))
        # Base
        base_w = int(25 * scale)
        draw.rectangle([center_x - base_w, stand_bottom_y, center_x + base_w, stand_bottom_y + int(8 * scale)], fill=stand_color)
        
    elif category == "Beauty":
        # Draw cosmetic bottle shapes (body + cap)
        bottle_color = (random.randint(220, 255), random.randint(100, 160), random.randint(160, 200))
        cap_color = (random.randint(180, 220), random.randint(150, 180), random.randint(50, 100))
        w = int(28 * scale)
        h = int(55 * scale)
        # Bottle
        draw.rectangle([center_x - w, center_y - h + int(20 * scale), center_x + w, center_y + h], fill=bottle_color)
        # Cap
        cap_w = int(18 * scale)
        draw.rectangle([center_x - cap_w, center_y - h, center_x + cap_w, center_y - h + int(20 * scale)], fill=cap_color)
        
    elif category == "Grocery":
        # Draw red apple / round vegetable
        apple_color = (random.randint(190, 240), random.randint(20, 60), random.randint(20, 60))
        stem_color = (random.randint(80, 120), random.randint(120, 180), random.randint(60, 100))
        r = int(45 * scale)
        # Two overlapping circles to look like an apple shape
        offset = int(12 * scale)
        draw.ellipse([center_x - r, center_y - r, center_x + offset, center_y + r], fill=apple_color)
        draw.ellipse([center_x - offset, center_y - r, center_x + r, center_y + r], fill=apple_color)
        # Stem
        draw.arc([center_x - int(10*scale), center_y - r - int(18*scale), center_x + int(10*scale), center_y - r], 
                 0, 180, fill=stem_color, width=max(2, int(4 * scale)))
        # Leaf
        draw.polygon([
            (center_x, center_y - r - int(10*scale)), 
            (center_x + int(15*scale), center_y - r - int(18*scale)), 
            (center_x + int(5*scale), center_y - r - int(5*scale))
        ], fill=stem_color)
        
    # Save the unique synthetic image
    img.save(output_path, "JPEG")

def generate_synthetic_dataset() -> None:
    """Creates a synthetic train/val dataset for testing model scripts."""
    logger.info("Generating synthetic dataset for testing...")
    
    # Recreate ONLY subdirectories, protecting the parent directory containing this script!
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)
    
    for cat in CATEGORIES:
        os.makedirs(os.path.join(TRAIN_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cat), exist_ok=True)
        
        # Generate 100 train images (all randomized/unique)
        for i in range(100):
            out_path = os.path.join(TRAIN_DIR, cat, f"train_{cat.lower()}_{i:03d}.jpg")
            create_synthetic_image(cat, out_path)
            
        # Generate 30 validation images (all randomized/unique)
        for i in range(30):
            out_path = os.path.join(VAL_DIR, cat, f"val_{cat.lower()}_{i:03d}.jpg")
            create_synthetic_image(cat, out_path)
            
    logger.info(f"Synthetic dataset created successfully at: {DATASET_DIR}")
    report_dataset_counts()

def map_directory_to_category(dir_name: str) -> str:
    """Maps a subfolder name to one of the 5 canonical categories."""
    dir_name_lower = dir_name.lower()
    
    # Direct match check
    for cat in CATEGORIES:
        if cat.lower() == dir_name_lower:
            return cat
            
    # Config-based map check
    for key, mapped_cat in CATEGORY_MAP.items():
        if key in dir_name_lower:
            return mapped_cat
            
    return None

def process_extracted_folder(source_dir: str) -> None:
    """Scans and parses an extracted folder, organizing it into canonical train/val split."""
    logger.info(f"Processing source folder: {source_dir}")
    
    # Initialize target folders
    for cat in CATEGORIES:
        os.makedirs(os.path.join(TRAIN_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cat), exist_ok=True)
        
    # Discover all image files and group by mapped category
    categorized_files: Dict[str, List[str]] = {cat: [] for cat in CATEGORIES}
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    for root, _, files in os.walk(source_dir):
        # Determine mapping based on the folder name
        folder_name = os.path.basename(root)
        category = map_directory_to_category(folder_name)
        
        # If folder doesn't match, check parent folder
        if not category:
            parent_folder_name = os.path.basename(os.path.dirname(root))
            category = map_directory_to_category(parent_folder_name)
            
        if not category:
            continue
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                categorized_files[category].append(file_path)
                
    logger.info("Found categorized images from source path:")
    for cat, file_list in categorized_files.items():
        logger.info(f"  {cat}: {len(file_list)} images found.")
        
    # Split and copy files (80/20 train/val split)
    for cat, file_list in categorized_files.items():
        if not file_list:
            logger.warning(f"No images found for category: {cat}. Visual evaluation will fail for this category.")
            continue
            
        random.shuffle(file_list)
        split_idx = int(len(file_list) * 0.8)
        train_files = file_list[:split_idx]
        val_files = file_list[split_idx:]
        
        # Clear existing images in target directories
        train_cat_dir = os.path.join(TRAIN_DIR, cat)
        val_cat_dir = os.path.join(VAL_DIR, cat)
        
        shutil.rmtree(train_cat_dir, ignore_errors=True)
        shutil.rmtree(val_cat_dir, ignore_errors=True)
        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(val_cat_dir, exist_ok=True)
        
        # Copy files with standard resizing and conversion to RGB JPEG
        def copy_and_format(src_paths: List[str], dest_dir: str, label: str) -> None:
            for idx, src in enumerate(src_paths):
                try:
                    with Image.open(src) as img:
                        # Convert to RGB if necessary
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Resize to config size
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        # Save
                        dest_path = os.path.join(dest_dir, f"{label}_{idx:05d}.jpg")
                        img.save(dest_path, "JPEG")
                except Exception as e:
                    logger.debug(f"Could not process image {src}: {e}")
                    
        copy_and_format(train_files, train_cat_dir, "train")
        copy_and_format(val_files, val_cat_dir, "val")
        
    logger.info("Image organization and splitting completed successfully.")
    report_dataset_counts()

def report_dataset_counts() -> None:
    """Analyzes the organized dataset and prints class distributions and balance checks."""
    logger.info("=== Dataset Distribution Report ===")
    
    train_counts = {}
    val_counts = {}
    
    for cat in CATEGORIES:
        t_dir = os.path.join(TRAIN_DIR, cat)
        v_dir = os.path.join(VAL_DIR, cat)
        
        t_count = len(os.listdir(t_dir)) if os.path.exists(t_dir) else 0
        v_count = len(os.listdir(v_dir)) if os.path.exists(v_dir) else 0
        
        train_counts[cat] = t_count
        val_counts[cat] = v_count
        
        logger.info(f"Category: {cat:12s} | Train: {t_count:5d} | Val: {v_count:5d}")
        
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    logger.info(f"Total Train Images: {total_train} | Total Val Images: {total_val}")
    
    if total_train > 0:
        # Check for class imbalance (max/min ratio)
        counts = [c for c in train_counts.values() if c > 0]
        if counts:
            max_c = max(counts)
            min_c = min(counts)
            ratio = max_c / min_c if min_c > 0 else float('inf')
            logger.info(f"Class imbalance ratio (Max/Min): {ratio:.2f}")
            if ratio > 2.0:
                logger.warning("Class imbalance exceeds 2:1! Class weights should be applied during model training.")
            else:
                logger.info("Class distribution is balanced (ratio <= 2:1).")

def download_from_kaggle(dataset_name: str) -> None:
    """Uses the Kaggle API to download and extract the dataset."""
    logger.info(f"Attempting to download Kaggle dataset: {dataset_name}")
    try:
        import kaggle
        # Create temp folder for download
        temp_dir = os.path.join(DATASET_DIR, "temp_kaggle")
        os.makedirs(temp_dir, exist_ok=True)
        
        kaggle.api.authenticate()
        logger.info("Kaggle API Authenticated successfully. Downloading dataset files...")
        
        kaggle.api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
        logger.info("Dataset downloaded and extracted. Organizing files...")
        
        process_extracted_folder(temp_dir)
        
        # Cleanup temp files
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.error("Please ensure your API token file (kaggle.json) is configured at ~/.kaggle/ or provide the dataset path manually using --path.")
        raise

# Mapping from kagglehub classes to canonical 5 classes
KAGGLER_CLASS_MAP = {
    "beauty_health": "Beauty",
    "clothing_accessories_jewellery": "Fashion",
    "electronics": "Electronics",
    "grocery": "Grocery",
    "home_kitchen_tools": "Home"
}

def process_kagglehub_dataset(source_dir: str, max_train_per_class: int = 400, max_val_per_class: int = 100) -> None:
    """Copies and formatting files from kagglehub directory structure into TRAIN_DIR and VAL_DIR."""
    logger.info(f"Processing kagglehub dataset from: {source_dir}")
    
    # Recreate subdirectories
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)
    
    for cat in CATEGORIES:
        os.makedirs(os.path.join(TRAIN_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cat), exist_ok=True)
        
    train_source = os.path.join(source_dir, "train")
    val_source = os.path.join(source_dir, "val")
    
    if not os.path.exists(train_source) or not os.path.exists(val_source):
        # Handle cases where ECOMMERCE_PRODUCT_IMAGES root is nested
        nested_dir = os.path.join(source_dir, "ECOMMERCE_PRODUCT_IMAGES")
        if os.path.exists(nested_dir):
            train_source = os.path.join(nested_dir, "train")
            val_source = os.path.join(nested_dir, "val")
            
    if not os.path.exists(train_source):
        logger.error(f"Train folder not found in source directory: {source_dir}")
        return
        
    def copy_split(src_root: str, dest_root: str, max_per_class: int, prefix: str):
        for subfolder in os.listdir(src_root):
            canonical_cat = KAGGLER_CLASS_MAP.get(subfolder.lower())
            if not canonical_cat:
                logger.info(f"Skipping directory '{subfolder}' (not in 5 canonical classes).")
                continue
                
            src_cat_dir = os.path.join(src_root, subfolder)
            dest_cat_dir = os.path.join(dest_root, canonical_cat)
            
            files = [f for f in os.listdir(src_cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
            random.shuffle(files)
            
            # Limit count
            files_to_copy = files[:max_per_class]
            logger.info(f"Copying {len(files_to_copy)} images for category '{canonical_cat}' ({prefix})...")
            
            for idx, file_name in enumerate(files_to_copy):
                src_file = os.path.join(src_cat_dir, file_name)
                dest_file = os.path.join(dest_cat_dir, f"{prefix}_{canonical_cat.lower()}_{idx:04d}.jpg")
                try:
                    with Image.open(src_file) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        img.save(dest_file, "JPEG")
                except Exception as e:
                    logger.debug(f"Failed to process {src_file}: {e}")
                    
    copy_split(train_source, TRAIN_DIR, max_train_per_class, "train")
    copy_split(val_source, VAL_DIR, max_val_per_class, "val")
    
    logger.info("Kagglehub dataset organization completed successfully.")
    report_dataset_counts()

def main() -> None:
    parser = argparse.ArgumentParser(description="Organize and setup dataset for SmartCatalog AI.")
    parser.add_argument("--path", type=str, help="Path to manual downloaded zip file or directory folder.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic image dataset for testing/CI.")
    parser.add_argument("--kagglehub", action="store_true", help="Download and extract the 18k Amazon dataset using kagglehub.")
    parser.add_argument("--dataset", type=str, default="suyashlakhani/ecommerce-products-image-dataset", 
                        help="Kaggle dataset identifier if pulling from Kaggle (default: suyashlakhani/ecommerce-products-image-dataset).")
    
    args = parser.parse_args()
    
    if args.synthetic:
        generate_synthetic_dataset()
        return
        
    if args.kagglehub:
        try:
            import kagglehub
            logger.info("Downloading fatihkgg/ecommerce-product-images-18k using kagglehub...")
            path = kagglehub.dataset_download("fatihkgg/ecommerce-product-images-18k")
            process_kagglehub_dataset(path)
        except Exception as e:
            logger.error(f"Kagglehub download failed: {e}")
        return
        
    if args.path:
        path = args.path
        if not os.path.exists(path):
            logger.error(f"Provided path does not exist: {path}")
            return
            
        if os.path.isfile(path) and zipfile.is_zipfile(path):
            logger.info(f"Extracting zip archive: {path}")
            temp_extract_dir = os.path.join(DATASET_DIR, "temp_zip_extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            process_extracted_folder(temp_extract_dir)
            shutil.rmtree(temp_extract_dir)
        elif os.path.isdir(path):
            process_extracted_folder(path)
        else:
            logger.error(f"Provided path is not a valid zip or directory: {path}")
        return
        
    # Check if kaggle package and configuration is available
    home_dir = os.path.expanduser("~")
    kaggle_config_path = os.path.join(home_dir, ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_config_path):
        try:
            download_from_kaggle(args.dataset)
            return
        except ImportError:
            logger.warning("Kaggle Python library is not installed. Run `pip install kaggle` to automate dataset download.")
            
    # Default fallback: check if we have a downloaded kagglehub folder in cache
    default_hub_path = os.path.join(home_dir, ".cache", "kagglehub", "datasets", "fatihkgg", "ecommerce-product-images-18k")
    if os.path.exists(default_hub_path):
        logger.info(f"Found downloaded kagglehub dataset in cache at {default_hub_path}. Organizing it...")
        # Search for ECOMMERCE_PRODUCT_IMAGES subfolders
        for root, dirs, _ in os.walk(default_hub_path):
            if "ECOMMERCE_PRODUCT_IMAGES" in dirs:
                path = os.path.join(root, "ECOMMERCE_PRODUCT_IMAGES")
                process_kagglehub_dataset(path)
                return
            elif "train" in dirs and "val" in dirs:
                process_kagglehub_dataset(root)
                return
                
    # If no path is provided and Kaggle download isn't available
    logger.error("No input dataset source could be resolved!")
    logger.info("To resolve this, please choose one of these actions:")
    logger.info("  1. Run with `--kagglehub` to download and organize the real 18k Amazon image dataset:")
    logger.info("     python dataset/download_dataset.py --kagglehub")
    logger.info("  2. Run with `--synthetic` to automatically generate a sample training dataset for validation/CI:")
    logger.info("     python dataset/download_dataset.py --synthetic")
    logger.info("  3. Manually download a product image dataset (e.g. from Kaggle) and pass its path:")
    logger.info("     python dataset/download_dataset.py --path C:\\path\\to\\dataset.zip")


if __name__ == "__main__":
    main()
