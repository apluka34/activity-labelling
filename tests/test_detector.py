"""
Tests for the YOLO detector module.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector.yolo_detector import YOLODetector


def test_detector_initialization():
    """Test that detector initialization raises error with invalid model path."""
    with pytest.raises(FileNotFoundError):
        YOLODetector("nonexistent_model.pt")


@pytest.mark.skipif(not os.path.exists("models/weights/yolo11x.pt"), 
                    reason="YOLO model not found")
def test_detector_detect():
    """Test detection on a sample image."""
    # Create a simple test image (black with white rectangle in middle)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (220, 220), (420, 420), (255, 255, 255), -1)
    
    # Initialize detector with existing model
    model_path = "models/weights/yolo11x.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    detector = YOLODetector(model_path)
    
    # Run detection
    detections = detector.detect(img)
    
    # We don't expect any real detections in this synthetic image,
    # but the function should run without errors
    assert isinstance(detections, list)


@pytest.mark.skipif(not os.path.exists("models/weights/yolo11x.pt"), 
                    reason="YOLO model not found")
def test_crop_detections():
    """Test cropping detections."""
    # Create a test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    
    # Create mock detections
    detections = [
        {
            "label": "person",
            "confidence": 0.9,
            "box": [100, 100, 300, 300]
        }
    ]
    
    # Initialize detector
    model_path = "models/weights/yolo11x.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    detector = YOLODetector(model_path)
    
    # Crop detections
    crops = detector.crop_detections(img, detections)
    
    # Check results
    assert len(crops) == 1
    assert crops[0][0].shape == (200, 200, 3)  # Crop dimensions
    assert crops[0][1] == detections[0]  # Detection info 