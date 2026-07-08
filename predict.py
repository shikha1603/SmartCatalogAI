import os
import json
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from typing import List, Tuple, Dict, Any, Union, Optional
import logging
from config import MODEL_PATH, CLASS_INDICES_PATH, CATEGORIES, DEVICE

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")

# Define the model architecture so it can be loaded
def get_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Creates the MobileNetV2 model with custom classification head."""
    if pretrained:
        # Load weights with modern torchvision API
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2()
        
    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )
    return model

# Global model cache to avoid reloading on every prediction in Streamlit
_MODEL_CACHE: Dict[str, Any] = {}

def load_inference_model() -> Tuple[nn.Module, List[str]]:
    """
    Loads the saved model and class mapping.
    
    Returns:
        A tuple of (loaded PyTorch model, list of category class names ordered by indices).
    """
    global _MODEL_CACHE
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["class_names"]
        
    logger.info(f"Loading inference model from {MODEL_PATH}...")
    
    # Load class indices mapping
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # Class indices is stored as {"0": "Fashion", "1": "Electronics", ...}
        # Sort keys to make sure the order matches indices
        sorted_keys = sorted(class_indices.keys(), key=lambda x: int(x))
        class_names = [class_indices[k] for k in sorted_keys]
    else:
        logger.warning(f"Class indices file not found at {CLASS_INDICES_PATH}. Defaulting to configuration categories.")
        class_names = CATEGORIES
        
    num_classes = len(class_names)
    model = get_model(num_classes=num_classes, pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        # Load weights on the target device
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        logger.info("Model weights loaded successfully.")
    else:
        logger.warning(f"No trained model found at {MODEL_PATH}. Prediction will return random outputs.")
        
    # Cache and return
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["class_names"] = class_names
    return model, class_names

def load_temperature() -> float:
    """Loads optimal scaling temperature T, defaulting to 1.0 if not calibrated."""
    calibration_file = os.path.join(os.path.dirname(MODEL_PATH), "calibration.json")
    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, "r") as f:
                data = json.load(f)
                return float(data.get("temperature", 1.0))
        except Exception as e:
            logger.warning(f"Error reading calibration temperature: {e}. Defaulting to 1.0.")
    return 1.0

# Define ImageNet normalization transforms for MobileNetV2
get_transform = lambda: transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_input: Union[str, Image.Image], model: Optional[nn.Module] = None, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Infers the category of a product image.
    
    Args:
        image_input: Either a file path string or a PIL Image object.
        model: Preloaded model. If None, loads from MODEL_PATH.
        top_k: Number of predictions to return.
        
    Returns:
        List of tuples: [(class_name, confidence_probability), ...] sorted descending.
    """
    try:
        # Load model and class names if not passed
        if model is None:
            model, class_names = load_inference_model()
        else:
            _, class_names = load_inference_model() # Retrieve class name mapping from cache/file
            
        # Open image if path is provided
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input
            
        # Ensure image is in RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Preprocess image
        transform = get_transform()
        tensor_img = transform(img).unsqueeze(0).to(DEVICE)  # Shape (1, 3, 224, 224)
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor_img)
            # Load temperature scaling parameter
            temp = load_temperature()
            if temp != 1.0:
                outputs = outputs / temp
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)  # Shape (num_classes,)
            
        # Convert to CPU numpy list
        probs = probabilities.cpu().numpy().tolist()
        
        # Zip class names with their probability
        pred_pairs = list(zip(class_names, probs))
        
        # Sort by probability descending
        pred_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return pred_pairs[:top_k]
        
    except Exception as e:
        logger.error(f"Error during image prediction: {e}")
        raise

