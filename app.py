import streamlit as st
import tempfile
import torch
from PIL import Image
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/plate_detector_v2/weights/best.pt', force_reload=True)

# Page layout
st.set_page_config(page_title="Car Plate Detection", layout="wide")
st.sidebar.title("Car Plate Detection")
st.sidebar.markdown("**Project App**")
uploaded_file = st.sidebar.file_uploader("Browse the file from your gallery", type=["jpg", "jpeg", "png"])

st.title("Project interface")
col1, col2 = st.columns(2)

if uploaded_file:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name

    # Load original image
    original_img = Image.open(img_path)
    col1.subheader("Original image")
    col1.image(original_img, use_container_width=True)

    # Run YOLOv5 detection
    results = model(img_path)
    results.render()
    detected_img = results.ims[0]  # Detected image with boxes

    # Display detected image
    col2.subheader("Detected image")
    col2.image(detected_img, channels="BGR", use_container_width=True)

