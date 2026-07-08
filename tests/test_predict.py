import unittest
import torch
import numpy as np
from PIL import Image
import os
import predict
from config import CATEGORIES

class TestPredict(unittest.TestCase):
    def test_get_model(self) -> None:
        """Verifies PyTorch model builds with the custom Dense head correctly."""
        model = predict.get_model(num_classes=5, pretrained=False)
        self.assertIsNotNone(model)
        self.assertTrue(isinstance(model, torch.nn.Module))
        
        # Check classification classifier head is replaced
        classifier = model.classifier
        self.assertEqual(len(classifier), 5) # Dropout, Linear, ReLU, Dropout, Linear
        self.assertTrue(isinstance(classifier[1], torch.nn.Linear))
        self.assertEqual(classifier[4].out_features, 5)

    def test_predict_image(self) -> None:
        """Verifies prediction pipeline returns ordered predictions and correct score ranges on dummy inputs."""
        # Create a dummy image
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        
        # Initialize a random model
        model = predict.get_model(num_classes=5, pretrained=False)
        model.eval()
        
        # Call predict
        predictions = predict.predict_image(img, model=model, top_k=3)
        
        # Assert structure
        self.assertEqual(len(predictions), 3)
        
        # Check output content
        classes_predicted = []
        confidences = []
        for label, conf in predictions:
            classes_predicted.append(label)
            confidences.append(conf)
            
            # Confidence should be a probability
            self.assertTrue(0.0 <= conf <= 1.0)
            
        # Verify classes are from the configured categories
        for name in classes_predicted:
            self.assertIn(name, CATEGORIES)
            
        # Verify they are sorted in descending order
        self.assertTrue(confidences[0] >= confidences[1])
        self.assertTrue(confidences[1] >= confidences[2])

if __name__ == "__main__":
    unittest.main()
