"""
Video processing utilities for lane detection
"""

import cv2
import numpy as np
from typing import Optional, Callable, List
from pathlib import Path
from tqdm import tqdm

class VideoProcessor:
    """Process videos for lane detection"""
    
    def __init__(self, detector: Callable):
        """
        Initialize video processor
        
        Args:
            detector: Function that takes image and returns detection results
        """
        self.detector = detector
    
    def process_video(self, input_path: str, output_path: str, 
                     show_progress: bool = True):
        """
        Process entire video
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            show_progress: Whether to show progress bar
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        processing_times = []
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            start_time = cv2.getTickCount()
            
            try:
                results = self.detector(frame)
                
                # Add FPS counter
                end_time = cv2.getTickCount()
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                processing_times.append(processing_time)
                
                current_fps = 1 / processing_time if processing_time > 0 else 0
                
                # Add FPS text to frame
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                          (width - 200, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)  # Write original frame if processing fails
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        
        if show_progress:
            pbar.close()
        
        # Print statistics
        if processing_times:
            avg_fps = 1 / np.mean(processing_times)
            print(f"\nProcessing completed!")
            print(f"Output saved to: {output_path}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total processing time: {np.sum(processing_times):.2f} seconds")
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_interval: Extract every nth frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                output_path = output_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames from {video_path}")
    
    def create_video_from_frames(self, frames_dir: str, output_path: str, 
                                fps: int = 30):
        """
        Create video from frames
        
        Args:
            frames_dir: Directory containing frames
            output_path: Path to save video
            fps: Frames per second
        """
        frames_dir = Path(frames_dir)
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        
        if not frame_paths:
            print(f"No frames found in {frames_dir}")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame_path in tqdm(frame_paths, desc="Creating video"):
            frame = cv2.imread(str(frame_path))
            out.write(frame)
        
        out.release()
        print(f"Video saved to: {output_path}")
