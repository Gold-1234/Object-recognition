import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import time

st.set_page_config(page_title="YOLOv8 Webcam Detection", layout="centered")
st.title("📸 Live Object Detection with YOLOv11")

model = YOLO("yolo11n.pt")  # You can replace with custom .pt model

start_button = st.button("Start Webcam Detection")

FRAME_WINDOW = st.image([])

if start_button:
    cap = cv2.VideoCapture(0)
    st.info("Press 'Stop' to end webcam")

    stop_button = st.button("Stop")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture image")
            break

        results = model(frame)[0]

        annotated_frame = results.plot()

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(annotated_frame)

    cap.release()
    st.success("Webcam stopped")