import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_recall_curve
import logging
from typing import Dict, Tuple, List

from config import (
    VAL_DIR, MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE, BATCH_SIZE, DEVICE,
    SAVED_MODEL_DIR
)
from train_model import CatalogDataset
from predict import load_inference_model, predict_image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("compute_thresholds")

CLASS_THRESHOLDS_PATH = os.path.join(SAVED_MODEL_DIR, "class_thresholds.json")

def optimize_thresholds(probs: np.ndarray, labels: np.ndarray, class_names: List[str], target_precision: float = 0.90) -> Dict[str, float]:
    """
    Finds the optimal threshold for each class targeting a validation precision of at least target_precision.
    Falls back to F1-score maximization if the precision target is unachievable.
    """
    thresholds = {}
    
    for i, class_name in enumerate(class_names):
        # Binary target: 1 if this class is the true label, else 0
        y_true_binary = (labels == i).astype(int)
        y_prob_class = probs[:, i]
        
        precisions, recalls, thresh_vals = precision_recall_curve(y_true_binary, y_prob_class)
        
        # Filter thresholds where precision >= target_precision
        valid_indices = np.where(precisions[:-1] >= target_precision)[0]
        
        if len(valid_indices) > 0:
            # We select the minimum threshold that guarantees the target precision
            opt_idx = valid_indices[0]
            opt_thresh = float(thresh_vals[opt_idx])
            logger.info(f"Class '{class_name}': Set threshold to {opt_thresh:.4f} (Validation Precision: {precisions[opt_idx]*100:.1f}%)")
        else:
            # Fallback: Maximize F1-score
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            opt_idx = np.argmax(f1_scores[:-1])
            opt_thresh = float(thresh_vals[opt_idx])
            logger.warning(f"Class '{class_name}': Precision target ({target_precision*100}%) unachievable. "
                           f"Falling back to Max F1 threshold: {opt_thresh:.4f} (Precision: {precisions[opt_idx]*100:.1f}%, Recall: {recalls[opt_idx]*100:.1f}%)")
                           
        # Clamp thresholds between 0.50 and 0.99 for practical safety
        thresholds[class_name] = max(0.50, min(0.99, opt_thresh))
        
    return thresholds

def run_simulation(probs: np.ndarray, labels: np.ndarray, class_names: List[str], 
                   global_thresh: float, per_class_thresholds: Dict[str, float]) -> None:
    """Simulates and compares global threshold routing vs per-class threshold routing."""
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    total = len(labels)
    
    # 1. Global Threshold Simulation
    global_approved = 0
    global_errors = 0
    
    for i in range(total):
        pred_idx = predictions[i]
        pred_class = class_names[pred_idx]
        conf = confidences[i]
        true_idx = labels[i]
        
        if conf >= global_thresh:
            global_approved += 1
            if pred_idx != true_idx:
                global_errors += 1
                
    global_approval_rate = (global_approved / total) * 100 if total > 0 else 0
    global_error_rate = (global_errors / global_approved) * 100 if global_approved > 0 else 0
    
    # 2. Per-class Threshold Simulation
    pc_approved = 0
    pc_errors = 0
    
    for i in range(total):
        pred_idx = predictions[i]
        pred_class = class_names[pred_idx]
        conf = confidences[i]
        true_idx = labels[i]
        
        thresh = per_class_thresholds.get(pred_class, global_thresh)
        
        if conf >= thresh:
            pc_approved += 1
            if pred_idx != true_idx:
                pc_errors += 1
                
    pc_approval_rate = (pc_approved / total) * 100 if total > 0 else 0
    pc_error_rate = (pc_errors / pc_approved) * 100 if pc_approved > 0 else 0
    
    print("\n" + "="*70)
    print("ROUTING THRESHOLD COMPARISON SIMULATION (Validation Partition)")
    print("="*70)
    print(f"{'Metric':35s} | {'Global ('+str(int(global_thresh*100))+'%)':14s} | {'Per-Class':14s}")
    print("-"*70)
    print(f"{'Total Validation Items':35s} | {total:14d} | {total:14d}")
    print(f"{'Auto-Approved Items':35s} | {global_approved:14d} | {pc_approved:14d}")
    print(f"{'Auto-Approval Rate (%)':35s} | {global_approval_rate:12.2f}% | {pc_approval_rate:12.2f}%")
    print(f"{'Auto-Approved Classification Errors':35s} | {global_errors:14d} | {pc_errors:14d}")
    print(f"{'Automation Error Rate (%)':35s} | {global_error_rate:12.2f}% | {pc_error_rate:12.2f}%")
    print("="*70)

def main() -> None:
    # 1. Setup validation loader
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(VAL_DIR) or len(os.listdir(VAL_DIR)) == 0:
        logger.error("Validation directory is missing or empty.")
        return
        
    val_dataset = CatalogDataset(VAL_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Check model
    if not os.path.exists(MODEL_PATH):
         logger.error("No trained model weights found. Please run train_model.py first.")
         return
         
    model, class_names = load_inference_model()
    model.eval()
    
    # 3. Collect scaled predictions
    probs_list = []
    labels_list = []
    
    from predict import load_temperature
    temp = load_temperature()
    logger.info(f"Accumulating validation predictions (calibrated temperature T = {temp:.4f})...")
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            if temp != 1.0:
                outputs = outputs / temp
            probs = torch.softmax(outputs, dim=1)
            probs_list.append(probs.cpu())
            labels_list.append(labels)
            
    probs = torch.cat(probs_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    # 4. Optimize per-class thresholds
    thresholds = optimize_thresholds(probs, labels, class_names, target_precision=0.90)
    
    # 5. Save thresholds to json
    os.makedirs(os.path.dirname(CLASS_THRESHOLDS_PATH), exist_ok=True)
    with open(CLASS_THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=4)
    logger.info(f"Saved optimized per-class thresholds to {CLASS_THRESHOLDS_PATH}")
    
    # 6. Run simulation comparison
    run_simulation(probs, labels, class_names, global_thresh=0.85, per_class_thresholds=thresholds)

if __name__ == "__main__":
    main()
