import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Any

from config import (
    VAL_DIR, MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE, BATCH_SIZE, DEVICE,
    OUTPUTS_DIR, SAVED_MODEL_DIR
)
from train_model import CatalogDataset
from predict import load_inference_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("calibrate")

CALIBRATION_JSON_PATH = os.path.join(SAVED_MODEL_DIR, "calibration.json")
RELIABILITY_PLOT_PATH = os.path.join(OUTPUTS_DIR, "reliability_diagram.png")

def calculate_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    
    Args:
        probs: Softmax probabilities of shape (N, num_classes).
        labels: True integer labels of shape (N,).
        num_bins: Number of confidence intervals to split predictions.
        
    Returns:
        The ECE score.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Identify elements belonging to the bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def get_reliability_bins(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Computes bin confidences and accuracies for reliability diagram."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            # If no items fall in bin, append defaults matching boundary center
            bin_accs.append(np.nan)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            
    return np.array(bin_confs), np.array(bin_accs)

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fits optimal scalar temperature parameter using PyTorch optimizer on validation logits."""
    # Temperature scaler parameter
    temperature = nn.Parameter(torch.ones(1, device=DEVICE) * 1.5)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize T using L-BFGS or Adam (Adam is simple and highly stable)
    optimizer = optim.Adam([temperature], lr=0.01)
    
    # Validation targets
    logits = logits.to(DEVICE)
    labels = labels.to(DEVICE)
    
    logger.info("Optimizing temperature scaling parameter T...")
    
    best_loss = float('inf')
    best_temp = 1.0
    
    for iteration in range(200):
        optimizer.zero_grad()
        # Scale logits
        scaled_logits = logits / temperature
        loss = criterion(scaled_logits, labels)
        loss.backward()
        optimizer.step()
        
        # Constrain temperature to be positive
        with torch.no_grad():
            temperature.clamp_(min=0.01)
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_temp = temperature.item()
            
    logger.info(f"Optimization completed. Optimal Temperature T = {best_temp:.4f} (Validation Loss: {best_loss:.6f})")
    return best_temp

def main() -> None:
    # 1. Setup validation loader
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(VAL_DIR) or len(os.listdir(VAL_DIR)) == 0:
        logger.error("Validation directory is missing or empty. Please run dataset setup first.")
        return
        
    val_dataset = CatalogDataset(VAL_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Check model
    if not os.path.exists(MODEL_PATH):
         logger.error(f"No trained model weights found at {MODEL_PATH}. Cannot calibrate.")
         return
         
    model, class_names = load_inference_model()
    model.eval()
    
    # 3. Collect logits & labels
    logits_list = []
    labels_list = []
    
    logger.info("Accumulating raw validation logits...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            logits_list.append(outputs.cpu())
            labels_list.append(labels)
            
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # 4. Compute pre-calibration stats
    probs_before = torch.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    
    ece_before = calculate_ece(probs_before, labels_np)
    logger.info(f"Uncalibrated Model Expected Calibration Error (ECE): {ece_before * 100:.2f}%")
    
    # 5. Fit temperature
    opt_temp = fit_temperature(logits, labels)
    
    # 6. Save optimal temperature parameters
    os.makedirs(os.path.dirname(CALIBRATION_JSON_PATH), exist_ok=True)
    with open(CALIBRATION_JSON_PATH, "w") as f:
        json.dump({"temperature": opt_temp}, f, indent=4)
    logger.info(f"Saved optimal calibration temperature to {CALIBRATION_JSON_PATH}")
    
    # 7. Compute post-calibration stats
    scaled_logits = logits / opt_temp
    probs_after = torch.softmax(scaled_logits, dim=1).numpy()
    
    ece_after = calculate_ece(probs_after, labels_np)
    logger.info(f"Calibrated Model Expected Calibration Error (ECE): {ece_after * 100:.2f}%")
    
    # 8. Plot Reliability Diagram
    bin_confs_before, bin_accs_before = get_reliability_bins(probs_before, labels_np)
    bin_confs_after, bin_accs_after = get_reliability_bins(probs_after, labels_np)
    
    plt.figure(figsize=(8, 8))
    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Before Calibration Plot
    # Mask NaNs for cleaner line representation
    mask_before = ~np.isnan(bin_accs_before)
    plt.plot(bin_confs_before[mask_before], bin_accs_before[mask_before], marker="o", color="#e74c3c", 
             label=f"Before Calibration (ECE: {ece_before*100:.1f}%)")
             
    # After Calibration Plot
    mask_after = ~np.isnan(bin_accs_after)
    plt.plot(bin_confs_after[mask_after], bin_accs_after[mask_after], marker="s", color="#0f766e", 
             label=f"After Calibration (T={opt_temp:.3f}, ECE: {ece_after*100:.1f}%)")
             
    plt.title("Confidence Calibration Reliability Diagram", fontsize=13, fontweight="bold")
    plt.xlabel("Average Predicted Confidence", fontsize=11)
    plt.ylabel("Actual Accuracy", fontsize=11)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(RELIABILITY_PLOT_PATH), exist_ok=True)
    plt.savefig(RELIABILITY_PLOT_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved reliability diagram plot to {RELIABILITY_PLOT_PATH}")

if __name__ == "__main__":
    main()
