import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import logging
from typing import List, Dict, Tuple, Any

from config import (
    VAL_DIR, MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE, BATCH_SIZE, DEVICE,
    ACCURACY_CURVE_PATH, LOSS_CURVE_PATH, CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH, OUTPUTS_DIR
)
from train_model import CatalogDataset
from predict import load_inference_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

def load_history() -> Optional[Dict[str, Any]]:
    """Loads saved training history JSON from saved_model folder."""
    history_file = os.path.join(os.path.dirname(MODEL_PATH), "history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return None

def plot_curves(history: Dict[str, Any]) -> None:
    """Generates and saves validation vs training loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    stage1_epochs = history.get("stage1_epochs", 0)
    
    # 1. Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [x * 100 for x in history["train_acc"]], 'o-', label='Train Accuracy', color='#1abc9c')
    plt.plot(epochs, [x * 100 for x in history["val_acc"]], 's-', label='Validation Accuracy', color='#e67e22')
    if stage1_epochs > 0:
        plt.axvline(x=stage1_epochs + 0.5, color='#7f8c8d', linestyle='--', label='Fine-Tuning Transition')
    plt.title('Training & Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(ACCURACY_CURVE_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved accuracy curve to {ACCURACY_CURVE_PATH}")
    
    # 2. Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], 'o-', label='Train Loss', color='#3498db')
    plt.plot(epochs, history["val_loss"], 's-', label='Validation Loss', color='#9b59b6')
    if stage1_epochs > 0:
        plt.axvline(x=stage1_epochs + 0.5, color='#7f8c8d', linestyle='--', label='Fine-Tuning Transition')
    plt.title('Training & Validation Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved loss curve to {LOSS_CURVE_PATH}")

def analyze_confusion(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Tuple[str, Tuple[str, str, int]]:
    """Dynamically finds the category pair with the highest number of misclassifications from the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create off-diagonal copy
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    
    # Get index of max confusion count
    max_idx = np.argmax(cm_off_diag)
    actual_idx, predicted_idx = np.unravel_index(max_idx, cm_off_diag.shape)
    count = cm_off_diag[actual_idx, predicted_idx]
    
    if count == 0:
        msg = "No misclassifications found! The model achieved perfect accuracy on the validation set."
        return msg, ("None", "None", 0)
        
    actual_class = class_names[actual_idx]
    predicted_class = class_names[predicted_idx]
    
    analysis = (
        f"Dynamic Confusion Analysis:\n"
        f"  - Most Confused Pair: Actual '{actual_class}' misclassified as Predicted '{predicted_class}'\n"
        f"  - Count: {count} occurrence(s)\n"
    )
    return analysis, (actual_class, predicted_class, int(count))

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str]) -> None:
    """Generates, plots, and saves both raw and normalized confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Clean divisions by zero if class is empty
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_title('Raw Confusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
    axes[1].set_title('Normalized Confusion Matrix', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    
    plt.suptitle('Model Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

def main() -> None:
    # Set transforms (matching training validation)
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check model existence
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Trained model not found at {MODEL_PATH}. Cannot perform evaluation. Please run train_model.py first.")
        return
        
    # Load model and class names
    model, class_names = load_inference_model()
    
    # Create dataset
    if not os.path.exists(VAL_DIR) or len(os.listdir(VAL_DIR)) == 0:
        logger.error(f"Validation dataset directory {VAL_DIR} is missing or empty.")
        return
        
    val_dataset = CatalogDataset(VAL_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Collect predictions
    y_true: List[int] = []
    y_pred: List[int] = []
    
    model.eval()
    logger.info("Evaluating model on validation dataset...")
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            
    # Generate sklearn classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_PATH)
    logger.info(f"Saved classification report to {CLASSIFICATION_REPORT_PATH}")
    
    # Print print-friendly console report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("="*60)
    
    # Generate confusion matrix plots
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Run dynamic confusion analysis
    analysis_text, (act, pred, cnt) = analyze_confusion(y_true, y_pred, class_names)
    print("\n" + "="*60)
    print(analysis_text)
    print("="*60)
    
    # Write confusion analysis metadata to files for app & README reference
    analysis_path = os.path.join(OUTPUTS_DIR, "confusion_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump({
            "analysis_text": analysis_text,
            "most_confused_actual": act,
            "most_confused_predicted": pred,
            "confusion_count": cnt
        }, f, indent=4)
        
    # Plot history curves
    history = load_history()
    if history:
        plot_curves(history)
    else:
        logger.warning("Training history.json file not found. Skipping loss/accuracy plots.")
        
    logger.info("Model evaluation completed.")

if __name__ == "__main__":
    main()
