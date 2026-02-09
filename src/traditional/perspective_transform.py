"""
Perspective transform utilities for lane detection
"""

import cv2
import numpy as np
from typing import Tuple

class PerspectiveTransformer:
    """Handles perspective transform operations"""
    
    def __init__(self, image_shape: Tuple[int, int]):
        """
        Initialize transformer
        
        Args:
            image_shape: (height, width) of input image
        """
        self.image_shape = image_shape
        self.height, self.width = image_shape
        
        # Define source and destination points
        self.src_points = self._get_source_points()
        self.dst_points = self._get_destination_points()
        
        # Calculate transformation matrices
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def _get_source_points(self) -> np.ndarray:
        """Get source points for perspective transform"""
        return np.float32([
            [self.width * 0.15, self.height * 0.9],    # Bottom-left
            [self.width * 0.45, self.height * 0.65],   # Top-left
            [self.width * 0.55, self.height * 0.65],   # Top-right
            [self.width * 0.85, self.height * 0.9]     # Bottom-right
        ])
    
    def _get_destination_points(self) -> np.ndarray:
        """Get destination points for perspective transform"""
        return np.float32([
            [self.width * 0.2, self.height],           # Bottom-left
            [self.width * 0.2, 0],                     # Top-left
            [self.width * 0.8, 0],                     # Top-right
            [self.width * 0.8, self.height]            # Bottom-right
        ])
    
    def warp(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transform (bird's eye view)"""
        return cv2.warpPerspective(image, self.M, (self.width, self.height), 
                                  flags=cv2.INTER_LINEAR)
    
    def unwarp(self, image: np.ndarray) -> np.ndarray:
        """Apply inverse perspective transform"""
        return cv2.warpPerspective(image, self.Minv, (self.width, self.height),
                                  flags=cv2.INTER_LINEAR)
    
    def draw_points(self, image: np.ndarray) -> np.ndarray:
        """Draw source points on image"""
        result = image.copy()
        
        for point in self.src_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(result, (x, y), 10, (0, 0, 255), -1)
        
        # Connect points
        cv2.polylines(result, [self.src_points.astype(np.int32)], True, 
                     (255, 0, 0), 2)
        
        return result
    
    def get_lane_points(self, left_fit: np.ndarray, right_fit: np.ndarray, 
                       ploty: np.ndarray) -> np.ndarray:
        """
        Get lane points for visualization
        
        Args:
            left_fit: Left lane polynomial coefficients
            right_fit: Right lane polynomial coefficients
            ploty: y-coordinates for plotting
            
        Returns:
            Array of points for lane polygon
        """
        # Generate x values from polynomials
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # Create array of points
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        return pts.astype(np.int32)
