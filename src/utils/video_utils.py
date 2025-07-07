"""
Utility functions for video processing.
"""
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from tqdm import tqdm


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with video information (fps, frame_count, width, height, duration)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval_seconds: float = 1.0,
    start_time: float = 0.0,
    end_time: float = None,
    output_format: str = "frame_{frame_number:06d}.jpg",
    show_progress: bool = True,
) -> List[str]:
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval_seconds: Time interval between frames to extract
        start_time: Start time in seconds
        end_time: End time in seconds (None for end of video)
        output_format: Format string for output filenames
        show_progress: Whether to show a progress bar
    
    Returns:
        List of paths to saved frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frame interval
    frame_interval = int(fps * frame_interval_seconds)
    if frame_interval < 1:
        frame_interval = 1
    
    # Set start and end frames
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames
    
    # Ensure valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    # Set initial position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    saved_paths = []
    frame_number = start_frame
    
    # Create progress bar if requested
    if show_progress:
        pbar = tqdm(
            total=(end_frame - start_frame) // frame_interval,
            desc="Extracting frames",
            unit="frames",
        )
    
    while frame_number < end_frame:
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame
        output_path = os.path.join(
            output_dir, output_format.format(frame_number=frame_number)
        )
        cv2.imwrite(output_path, frame)
        saved_paths.append(output_path)
        
        # Update progress
        if show_progress:
            pbar.update(1)
        
        # Skip to next frame
        frame_number += frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Close progress bar
    if show_progress:
        pbar.close()
    
    # Release video capture
    cap.release()
    
    return saved_paths


def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    frame_pattern: str = "frame_*.jpg",
    codec: str = "mp4v",
    sort_frames: bool = True,
) -> bool:
    """
    Create a video from a sequence of frames.
    
    Args:
        frames_dir: Directory containing the frames
        output_path: Path to save the output video
        fps: Frames per second for the output video
        frame_pattern: Pattern to match frame files
        codec: FourCC codec code
        sort_frames: Whether to sort the frames by name
    
    Returns:
        True if successful, False otherwise
    """
    import glob
    
    # Find all matching frames
    frame_paths = glob.glob(os.path.join(frames_dir, frame_pattern))
    if not frame_paths:
        print(f"No frames found matching pattern {frame_pattern} in {frames_dir}")
        return False
    
    # Sort frames if requested
    if sort_frames:
        frame_paths.sort()
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_paths[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Could not open output video file: {output_path}")
        return False
    
    # Write frames to video
    for frame_path in tqdm(frame_paths, desc="Creating video", unit="frames"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    
    # Release video writer
    out.release()
    
    return True 