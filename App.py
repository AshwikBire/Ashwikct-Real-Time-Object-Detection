import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import platform
import os
from datetime import datetime
import pandas as pd
import time

# 🌐 App config
st.set_page_config(page_title="Ashwik's Object Detection", layout="wide")
st.title("🚀 Ashwik's Real-Time Object Detection")
st.caption("Built with Streamlit, OpenCV, PyTorch")

# Sidebar Config
confidence_threshold = st.sidebar.slider("🔍 Confidence Threshold", 0.1, 1.0, 0.3)
model_type = st.sidebar.selectbox("📦 YOLOv5 Model", ["yolov5s", "yolov5m", "yolov5l", "yolov5x"])
selected_mode = st.sidebar.radio("🎥 Detection Mode", ["Webcam", "Upload Image", "Upload Video"])
alert_classes_input = st.sidebar.text_input("🔔 Alert Classes (comma-separated)", value="person,car")
dark_mode = st.sidebar.checkbox("🌗 Dark Mode")
show_fps = st.sidebar.checkbox("📈 Show FPS Counter")
show_log = st.sidebar.checkbox("📄 Show Detection Logs")
export_log = st.sidebar.button("📁 Export Logs to CSV")

alert_classes = [cls.strip().lower() for cls in alert_classes_input.split(",")]

# Apply theme
if dark_mode:
    st.markdown("""<style>body { background-color: #111; color: white }</style>""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model(model_name):
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    return model

model = load_model(model_type)
model.conf = confidence_threshold

# Beep for alerts
def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 300)
    else:
        os.system('play -nq -t alsa synth 0.3 sine 1000')

# Logging
log_file = "detection_log.csv"
def log_detection(label):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a") as f:
        f.write(f"{now},{label}\n")

# Upload handling
uploaded_file = None
if selected_mode != "Webcam":
    uploaded_file = st.file_uploader("📁 Upload Image or Video", type=['jpg', 'png', 'jpeg', 'mp4'])

# FPS calculation
def calculate_fps(start, end):
    return round(1 / (end - start), 2)

# Main Detection Logic
def detect_objects(source):
    stframe = st.empty()
    snapshot_btn = st.button("📸 Take Snapshot")
    cap = cv2.VideoCapture(source if isinstance(source, str) else 0)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        labels = results.pandas().xyxy[0]['name']
        label_counts = labels.value_counts().to_dict()

        # Render detection on frame
        results.render()
        img_out = Image.fromarray(results.ims[0])

        # Alerts
        for label in labels:
            log_detection(label)
            if label.lower() in alert_classes:
                beep()

        # Sidebar: object counts
        st.sidebar.markdown("### 🧠 Detected:")
        for label, count in label_counts.items():
            st.sidebar.write(f"🔹 {label}: {count}")

        # FPS
        end_time = time.time()
        fps = calculate_fps(start_time, end_time) if show_fps else None

        # Show image
        if show_fps:
            stframe.markdown(f"⏱️ FPS: `{fps}`")
        stframe.image(img_out, use_column_width=True)

        # Snapshot
        if snapshot_btn:
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_out.save(f"snapshot_{now}.jpg")
            st.success(f"📸 Snapshot saved: snapshot_{now}.jpg")

        if st.button("⛔ Stop"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Mode: Webcam
if selected_mode == "Webcam" and st.button("▶️ Start Webcam"):
    detect_objects(0)

# Mode: Upload Image
if selected_mode == "Upload Image" and uploaded_file:
    img = Image.open(uploaded_file)
    results = model(img)
    results.render()
    st.image(Image.fromarray(results.ims[0]), caption="🖼️ Detected Image", use_column_width=True)

# Mode: Upload Video
if selected_mode == "Upload Video" and uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    detect_objects("temp_video.mp4")

# Show Logs
if show_log and os.path.exists(log_file):
    df_logs = pd.read_csv(log_file, names=["Time", "Object"])
    st.dataframe(df_logs.tail(20))

# Export Logs
if export_log and os.path.exists(log_file):
    st.download_button("📥 Download CSV", open(log_file, "rb"), file_name="detection_log.csv")
