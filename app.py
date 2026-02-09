"""
Main entry point for Lane Detection System
"""

import argparse
import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.traditional.hough_detector import HoughLaneDetector
from src.traditional.sliding_window import SlidingWindowDetector
from src.deep_learning.predictor import DeepLearningPredictor
from src.utils.visualization import LaneVisualizer
from src.utils.metrics import calculate_metrics
from src.config import config

class LaneDetectionSystem:
    """Main system class"""
    
    def __init__(self, method: str = "traditional"):
        """
        Initialize lane detection system
        
        Args:
            method: Detection method ('traditional', 'deep_learning', 'hybrid')
        """
        self.method = method
        self.visualizer = LaneVisualizer()
        
        # Initialize detectors based on method
        if method == "traditional":
            self.detector = HoughLaneDetector()
        elif method == "sliding_window":
            self.detector = SlidingWindowDetector()
        elif method == "deep_learning":
            self.detector = DeepLearningPredictor()
        elif method == "hybrid":
            self.detectors = {
                'traditional': HoughLaneDetector(),
                'deep_learning': DeepLearningPredictor()
            }
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def process_image(self, image_path: str, output_path: Optional[str] = None):
        """
        Process single image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
            
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Process based on method
        if self.method == "hybrid":
            results = {}
            for name, detector in self.detectors.items():
                print(f"\nRunning {name} detector...")
                results[name] = detector.detect(image.copy())
        else:
            results = self.detector.detect(image)
            
        # Visualize results
        if self.method == "hybrid":
            # Create comparison visualization
            images = [image.copy()]
            titles = ["Original"]
            
            for name, result in results.items():
                if 'left_lane' in result:  # Traditional method
                    visualized = self.visualizer.draw_lanes(
                        image.copy(), 
                        result['left_lane'], 
                        result['right_lane']
                    )
                    visualized = self.visualizer.draw_curvature_text(
                        visualized,
                        result.get('curvature', 0),
                        result.get('offset', 0)
                    )
                else:  # Deep learning method
                    visualized = result.get('visualization', image.copy())
                    
                images.append(visualized)
                titles.append(name.replace('_', ' ').title())
                
            fig = self.visualizer.create_comparison_plot(images, titles)
            
            if output_path:
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Comparison saved to: {output_path}")
                
        else:
            # Single method visualization
            if 'left_lane' in results:  # Traditional method
                visualized = self.visualizer.draw_lanes(
                    image, 
                    results['left_lane'], 
                    results['right_lane']
                )
                visualized = self.visualizer.draw_lane_area(
                    visualized,
                    results['left_lane'],
                    results['right_lane']
                )
                visualized = self.visualizer.draw_curvature_text(
                    visualized,
                    results.get('curvature', 0),
                    results.get('offset', 0)
                )
            else:  # Deep learning method
                visualized = results.get('visualization', image)
                
            # Display or save
            if output_path:
                cv2.imwrite(output_path, visualized)
                print(f"Result saved to: {output_path}")
            else:
                cv2.imshow("Lane Detection", visualized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        return results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process video file
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        # Initialize video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every nth frame for performance
            if frame_count % 2 == 0:  # Process every 2nd frame
                continue
                
            print(f"Processing frame {frame_count}...")
            
            # Start timing
            start_time = cv2.getTickCount()
            
            # Process frame
            results = self.detector.detect(frame)
            
            # Calculate processing time
            end_time = cv2.getTickCount()
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            processing_times.append(processing_time)
            
            # Visualize
            if 'left_lane' in results:
                visualized = self.visualizer.draw_lanes(
                    frame, 
                    results['left_lane'], 
                    results['right_lane']
                )
                visualized = self.visualizer.draw_lane_area(
                    visualized,
                    results['left_lane'],
                    results['right_lane']
                )
                visualized = self.visualizer.draw_curvature_text(
                    visualized,
                    results.get('curvature', 0),
                    results.get('offset', 0)
                )
            else:
                visualized = results.get('visualization', frame)
                
            # Add FPS counter
            fps_text = f"FPS: {1/processing_time:.1f}"
            cv2.putText(visualized, fps_text, (width - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write or display
            if output_path:
                out.write(visualized)
            else:
                cv2.imshow("Lane Detection", visualized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        # Cleanup
        cap.release()
        if output_path:
            out.release()
            print(f"Processed video saved to: {output_path}")
        cv2.destroyAllWindows()
        
        # Print statistics
        if processing_times:
            avg_fps = len(processing_times) / sum(processing_times)
            print(f"\nProcessing Statistics:")
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average processing time: {1000/avg_fps:.2f} ms")
            
    def process_webcam(self):
        """
        Real-time webcam processing
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Starting webcam lane detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = self.detector.detect(frame)
            
            # Visualize
            if 'left_lane' in results:
                visualized = self.visualizer.draw_lanes(
                    frame, 
                    results['left_lane'], 
                    results['right_lane']
                )
                visualized = self.visualizer.draw_curvature_text(
                    visualized,
                    results.get('curvature', 0),
                    results.get('offset', 0)
                )
            else:
                visualized = results.get('visualization', frame)
                
            # Display
            cv2.imshow("Lane Detection - Webcam", visualized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Lane Detection System for Data Science Pinnacle"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['image', 'video', 'webcam', 'compare', 'train'],
        help='Processing mode'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (for image/video mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (optional)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='traditional',
        choices=['traditional', 'sliding_window', 'deep_learning', 'hybrid'],
        help='Detection method'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset path for training/comparison'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs for training'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize system
    print("=" * 60)
    print("LANE DETECTION SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Method: {args.method}")
    
    system = LaneDetectionSystem(method=args.method)
    
    # Execute based on mode
    try:
        if args.mode == 'image':
            if not args.input:
                print("Error: --input required for image mode")
                return
            system.process_image(args.input, args.output)
            
        elif args.mode == 'video':
            if not args.input:
                print("Error: --input required for video mode")
                return
            system.process_video(args.input, args.output)
            
        elif args.mode == 'webcam':
            system.process_webcam()
            
        elif args.mode == 'train':
            from src.deep_learning.trainer import train_model
            if not args.dataset:
                print("Error: --dataset required for training")
                return
            train_model(args.dataset, epochs=args.epochs)
            
        elif args.mode == 'compare':
            from src.utils.comparison import compare_methods
            if not args.dataset:
                print("Error: --dataset required for comparison")
                return
            compare_methods(args.dataset, output_path=args.output)
            
        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
