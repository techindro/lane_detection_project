import streamlit as st
import cv2
from src.lane_detector import LaneDetector
import numpy as np
from PIL import Image

st.title("🚗 Lane Detection Demo")
st.write("Upload video/image for real-time lane detection")

detector = LaneDetector()

uploaded_file = st.file_uploader("Choose video...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        result = detector.process_frame(frame)
        stframe.image(result, channels="BGR")
    
    cap.release()
