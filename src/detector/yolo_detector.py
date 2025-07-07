"""
YOLO-based human detector module.
"""
import os
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    A class for detecting objects in images using YOLO models.
    Primarily focused on human detection with configurable parameters.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: List[str] = None,
        padding: int = 20,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            target_classes: List of class names to detect (None for all classes)
            padding: Number of pixels to expand the bounding box in all directions
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []
        self.padding = padding

    def detect(
        self, image: np.ndarray
    ) -> List[Dict[str, Union[str, float, List[int]]]]:
        """
        Perform object detection on the given image.

        Args:
            image: Input image as a numpy array (BGR format from OpenCV)

        Returns:
            A list of detected objects, each as a dictionary containing
            label, confidence, and padded bounding box coordinates.
        """
        height, width = image.shape[:2]

        # Perform inference with configurable thresholds
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=640,
        )

        result = results[0]
        objects = []

        for xyxy, conf, cls in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            label = self.model.names[int(cls)]

            # Filter by target classes if specified
            if self.target_classes and label not in self.target_classes:
                continue

            # Extract coordinates
            left, top, right, bottom = xyxy.cpu().numpy().astype(int)

            # Apply padding, ensuring we stay within image bounds
            left_padded = max(left - self.padding, 0)
            top_padded = max(top - self.padding, 0)
            right_padded = min(right + self.padding, width - 1)
            bottom_padded = min(bottom + self.padding, height - 1)

            # Append result
            objects.append(
                {
                    "label": label,
                    "confidence": conf.item(),
                    "box": [left_padded, top_padded, right_padded, bottom_padded],
                }
            )

        return objects

    def crop_detections(
        self, image: np.ndarray, detections: List[Dict], min_confidence: float = None
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Crop detected objects from the image.

        Args:
            image: Input image as a numpy array
            detections: List of detection dictionaries from detect()
            min_confidence: Minimum confidence threshold (overrides the detector's threshold)

        Returns:
            List of tuples containing (cropped_image, detection_info)
        """
        threshold = min_confidence if min_confidence is not None else self.conf_threshold
        crops = []

        for det in detections:
            if det["confidence"] < threshold:
                continue

            x1, y1, x2, y2 = det["box"]
            cropped = image[y1:y2, x1:x2]
            
            # Ensure the crop is not empty
            if cropped.size > 0:
                crops.append((cropped, det))

        return crops 