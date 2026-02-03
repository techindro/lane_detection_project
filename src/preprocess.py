import cv2
import numpy as np
from .utils import region_of_interest, canny_edge

def preprocess_frame(frame):
    height, width = frame.shape[:2]
    
    # ROI vertices (trapezoid for road)
    vertices = np.array([[
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    
    # Pipeline
    edges = canny_edge(frame)
    roi = region_of_interest(edges, vertices)
    return roi
