import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional

from config import (
    TRAIN_DIR, VAL_DIR, MODEL_PATH, CLASS_INDICES_PATH,
    IMAGE_SIZE, BATCH_SIZE, DEVICE, CATEGORIES,
    EPOCHS_STAGE1, EPOCHS_STAGE2, LR_STAGE1, LR_STAGE2
)
from predict import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")

class CatalogDataset(Dataset):
    """Custom PyTorch dataset for product catalog images."""
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Determine class names and mapping
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples: List[Tuple[str, int]] = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file in os.listdir(cls_dir):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    self.samples.append((os.path.join(cls_dir, file), self.class_to_idx[cls_name]))
                    
        logger.info(f"Loaded {len(self.samples)} samples across {len(self.classes)} classes from {root_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            # Return a dummy blank image in case of corruption
            img = Image.new("RGB", IMAGE_SIZE, color=(255, 255, 255))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def compute_class_weights(dataset: CatalogDataset) -> Optional[torch.Tensor]:
    """Calculates class weights to handle training class imbalance if ratio > 2:1."""
    labels = [sample[1] for sample in dataset.samples]
    num_classes = len(dataset.classes)
    counts = np.bincount(labels, minlength=num_classes)
    
    max_c = max(counts)
    min_c = min(counts) if min(counts) > 0 else 1
    imbalance_ratio = max_c / min_c
    
    logger.info(f"Class distribution: {dict(zip(dataset.classes, counts))}")
    
    if imbalance_ratio > 2.0:
        logger.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.2f} > 2.0). Computing class weights.")
        total_samples = len(labels)
        # Standard formula: total_samples / (num_classes * class_count)
        weights = [total_samples / (num_classes * max(c, 1)) for c in counts]
        return torch.FloatTensor(weights).to(DEVICE)
    else:
        logger.info(f"Dataset is balanced (ratio: {imbalance_ratio:.2f} <= 2.0). Class weighting is not required.")
        return None

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Evaluates the model on validation data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main() -> None:
    # 1. Image Augmentations suitable for product images
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Check dataset directories
    if not os.path.exists(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
        logger.error(f"Training dataset directory {TRAIN_DIR} is missing or empty. Please run `python dataset/download_dataset.py` first.")
        return
        
    # 3. Create datasets and dataloaders
    train_dataset = CatalogDataset(TRAIN_DIR, train_transform)
    val_dataset = CatalogDataset(VAL_DIR, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    
    # Save class indices mapping
    class_to_idx_inv = {i: c for c, i in train_dataset.class_to_idx.items()}
    os.makedirs(os.path.dirname(CLASS_INDICES_PATH), exist_ok=True)
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(class_to_idx_inv, f, indent=4)
    logger.info(f"Saved class index mapping to {CLASS_INDICES_PATH}")
    
    # Determine Loss Criterion & Class Weights
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Build Model
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(DEVICE)
    
    # STAGE 1: Train dense head (base model frozen)
    logger.info("==========================================")
    logger.info("STAGE 1: Training Dense Classification Head")
    logger.info("==========================================")
    
    # Freeze MobileNetV2 base layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Optimizer for head
    optimizer_s1 = optim.Adam(model.classifier.parameters(), lr=LR_STAGE1)
    
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    for epoch in range(1, EPOCHS_STAGE1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_s1)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logger.info(f"Epoch {epoch:02d}/{EPOCHS_STAGE1:02d} | "
                    f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("--> Checkpoint saved (val_loss decreased).")
            
    stage1_best_acc = max(history["val_acc"])
    logger.info(f"Stage 1 completed. Best Val Accuracy: {stage1_best_acc * 100:.2f}%")
    
    # Load best checkpoint weights before Stage 2
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        
    # STAGE 2: Fine-tuning last ~30-40 layers of MobileNetV2 base
    logger.info("==========================================")
    logger.info("STAGE 2: Fine-tuning Last Features Layers")
    logger.info("==========================================")
    
    # Unfreeze block 14 to 18 of MobileNetV2 (each block contains bottleneck layers)
    # model.features has 19 elements (0 to 18)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[14:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Verify grad status
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} out of {total_params:,}")
    
    # Stage 2 optimizer with lower learning rate
    optimizer_s2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE2)
    
    # Callback metrics
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, mode='min', factor=0.5, patience=2)
    
    patience = 4
    no_improve_epochs = 0
    best_val_loss = float("inf")
    
    stage2_history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    for epoch in range(1, EPOCHS_STAGE2 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_s2)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Step scheduler
        lr_scheduler.step(val_loss)
        current_lr = optimizer_s2.param_groups[0]['lr']
        
        stage2_history["train_loss"].append(tr_loss)
        stage2_history["train_acc"].append(tr_acc)
        stage2_history["val_loss"].append(val_loss)
        stage2_history["val_acc"].append(val_acc)
        
        logger.info(f"Epoch {epoch:02d}/{EPOCHS_STAGE2:02d} | LR: {current_lr:.2e} | "
                    f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        
        # Checkpoint save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("--> Checkpoint saved (val_loss decreased).")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        # Early Stopping check
        if no_improve_epochs >= patience:
            logger.warning(f"Early stopping triggered after {epoch} epochs of Stage 2.")
            break
            
    # Combine training history for plotting later in evaluation
    combined_history = {
        "train_loss": history["train_loss"] + stage2_history["train_loss"],
        "train_acc": history["train_acc"] + stage2_history["train_acc"],
        "val_loss": history["val_loss"] + stage2_history["val_loss"],
        "val_acc": history["val_acc"] + stage2_history["val_acc"],
        "stage1_epochs": len(history["train_loss"])
    }
    
    # Save training history stats to outputs dir for graphing
    history_file = os.path.join(os.path.dirname(MODEL_PATH), "history.json")
    with open(history_file, "w") as f:
        json.dump(combined_history, f, indent=4)
        
    stage2_best_acc = max(stage2_history["val_acc"]) if stage2_history["val_acc"] else stage1_best_acc
    logger.info("==========================================")
    logger.info("Training Pipeline Completed Successfully.")
    logger.info(f"Stage 1 Best Val Acc: {stage1_best_acc * 100:.2f}%")
    logger.info(f"Stage 2 Best Val Acc: {stage2_best_acc * 100:.2f}%")
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info("==========================================")

if __name__ == "__main__":
    main()
