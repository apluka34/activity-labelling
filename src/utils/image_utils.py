"""
Utility functions for image processing.
"""
import os
from typing import Tuple, Optional, Dict, List, Union

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Preprocess an image to the ideal specifications for YOLO.
    
    Args:
        image: Input image as a numpy array (BGR format from OpenCV)
        target_size: Target size as (width, height), should be multiples of 32
                    Default is 640x640 which is optimal for YOLOv11
    
    Returns:
        Preprocessed image in BGR format
    """
    # Validate target size (both dimensions should be multiples of 32)
    if target_size[0] % 32 != 0 or target_size[1] % 32 != 0:
        print(f"Warning: Target size {target_size} is not a multiple of 32. This may affect model performance.")
    
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Create a black canvas of target size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate scaling factor to preserve aspect ratio
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate position to paste (center)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    # Place the resized image on the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize an image to the target dimensions without padding.
    
    Args:
        image: Input image as a numpy array (BGR format from OpenCV)
        target_size: Target size as (width, height)
    
    Returns:
        Resized image in BGR format
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array in BGR format, or None if loading fails
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file, creating directories if needed.
    
    Args:
        image: Image as numpy array in BGR format
        output_path: Path where to save the image
    
    Returns:
        True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        return cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Union[str, float, List[int]]]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    min_confidence: float = None,
) -> np.ndarray:
    """
    Draw detection boxes and labels on an image.
    
    Args:
        image: Input image as a numpy array (BGR format from OpenCV)
        detections: List of detection dictionaries from detect()
        color: BGR color tuple for the bounding box
        thickness: Line thickness for the bounding box
        font_scale: Font scale for the label text
        min_confidence: Minimum confidence threshold for showing detections
    
    Returns:
        Image with drawn detections
    """
    # Make a copy to avoid modifying the original
    result_image = image.copy()
    
    for det in detections:
        if min_confidence is not None and det["confidence"] < min_confidence:
            continue
            
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        text = f"{label}: {conf:.2f}"
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            result_image,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )
        
        # Draw text
        cv2.putText(
            result_image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
    
    return result_image 