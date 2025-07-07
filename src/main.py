#!/usr/bin/env python3
"""
Human Activity Detection System - Main Entry Point

This script processes video files to detect and extract human figures
using YOLO object detection.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector.yolo_detector import YOLODetector
from src.utils.image_utils import preprocess_image, resize_image, save_image, draw_detections
from src.utils.video_utils import get_video_info, extract_frames


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return {}


def process_video(
    video_path: str,
    output_dir: str,
    model_path: str,
    config: Dict = None,
    frame_interval_seconds: float = None,
    conf_threshold: float = None,
    iou_threshold: float = None,
    padding: int = None,
    target_size: Tuple[int, int] = None,
    save_original_frames: bool = None,
    save_visualization: bool = None,
) -> List[Dict]:
    """
    Process a video file to detect and extract human figures.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save output files
        model_path: Path to the YOLO model weights
        config: Configuration dictionary
        frame_interval_seconds: Time interval between frames to process
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        padding: Padding around detected objects
        target_size: Target size for output images
        save_original_frames: Whether to save original frames
        save_visualization: Whether to save visualization with bounding boxes
        
    Returns:
        List of dictionaries with detection results
    """
    # Use provided config or empty dict
    config = config or {}
    
    # Extract parameters from config with overrides from function arguments
    model_config = config.get("model", {})
    detection_config = config.get("detection", {})
    video_config = config.get("video", {})
    output_config = config.get("output", {})
    
    # Set parameters with priority: function args > config > defaults
    frame_interval = frame_interval_seconds or video_config.get("frame_interval_seconds", 3)
    confidence = conf_threshold or model_config.get("confidence_threshold", 0.5)
    iou = iou_threshold or model_config.get("iou_threshold", 0.45)
    box_padding = padding or detection_config.get("padding", 20)
    target_classes = detection_config.get("target_classes", ["person"])
    
    save_frames = save_original_frames if save_original_frames is not None else output_config.get("save_original_frames", True)
    save_vis = save_visualization if save_visualization is not None else output_config.get("save_visualization", False)
    
    # Create output directories
    crops_dir = os.path.join(output_dir, "crops")
    frames_dir = os.path.join(output_dir, "frames")
    vis_dir = os.path.join(output_dir, "visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize detector
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=confidence,
        iou_threshold=iou,
        target_classes=target_classes,
        padding=box_padding,
    )
    
    # Get video info
    video_info = get_video_info(video_path)
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"  Duration: {video_info['duration']:.2f}s")
    print(f"  Frames: {video_info['frame_count']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  Sampling interval: {frame_interval}s")
    
    # Calculate frame interval in frames
    frame_interval_frames = int(video_info["fps"] * frame_interval)
    if frame_interval_frames < 1:
        frame_interval_frames = 1
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Prepare for processing
    results = []
    frame_number = 0
    crop_counter = 0
    
    # Get naming format from config
    naming_format = output_config.get("naming_format", "frame{frame_number:06d}_det{detection_index:02d}")
    
    # Process frames
    with tqdm(total=video_info["frame_count"] // frame_interval_frames, desc="Processing frames") as pbar:
        while frame_number < video_info["frame_count"]:
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_number}")
                frame_number += frame_interval_frames
                pbar.update(1)
                continue
            
            # Get timestamp
            timestamp = frame_number / video_info["fps"]
            
            # Save original frame if requested
            if save_frames:
                frame_path = os.path.join(frames_dir, f"frame_{frame_number:06d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # Run detection
            detections = detector.detect(frame)
            
            # Save visualization if requested
            if save_vis and detections:
                vis_frame = draw_detections(frame, detections)
                vis_path = os.path.join(vis_dir, f"vis_frame_{frame_number:06d}.jpg")
                cv2.imwrite(vis_path, vis_frame)
            
            # Process each detection
            for i, det in enumerate(detections):
                # Get bounding box
                x1, y1, x2, y2 = det["box"]
                
                # Crop the detection
                cropped = frame[y1:y2, x1:x2]
                
                # Skip empty crops
                if cropped.size == 0:
                    continue
                
                # Resize if target size is specified
                if target_size:
                    cropped = resize_image(cropped, target_size)
                
                # Generate output path
                crop_path = os.path.join(
                    crops_dir,
                    f"{naming_format.format(frame_number=frame_number, detection_index=i)}.jpg"
                )
                
                # Save cropped image
                cv2.imwrite(crop_path, cropped)
                crop_counter += 1
                
                # Add to results
                results.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "detection_index": i,
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "crop_path": crop_path,
                })
            
            # Move to next frame
            frame_number += frame_interval_frames
            pbar.update(1)
    
    # Release video
    cap.release()
    
    print(f"Processing complete: {crop_counter} human detections saved to {crops_dir}")
    
    return results


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Human Activity Detection System - Detect and extract human figures from videos"
    )
    parser.add_argument(
        "--video", "-v", required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--model", "-m", help="Path to YOLO model weights", default="models/weights/yolo11x.pt"
    )
    parser.add_argument(
        "--config", "-c", help="Path to configuration file", default="config/default_config.yaml"
    )
    parser.add_argument(
        "--interval", "-i", type=float, help="Frame sampling interval in seconds"
    )
    parser.add_argument(
        "--conf", type=float, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--padding", "-p", type=int, help="Padding around detections in pixels"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save original frames"
    )
    parser.add_argument(
        "--save-vis", action="store_true", help="Save visualization with bounding boxes"
    )
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Load configuration
    config = load_config(args.config)
    
    # Process video
    try:
        start_time = time.time()
        results = process_video(
            video_path=args.video,
            output_dir=args.output,
            model_path=args.model,
            config=config,
            frame_interval_seconds=args.interval,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            padding=args.padding,
            save_original_frames=args.save_frames,
            save_visualization=args.save_vis,
        )
        elapsed = time.time() - start_time
        
        print(f"Total processing time: {elapsed:.2f}s")
        print(f"Detected {len(results)} human instances")
        
        # Save results to JSON file
        if results:
            import json
            results_path = os.path.join(args.output, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 