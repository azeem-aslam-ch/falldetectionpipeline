import sys
import os
import site
import time
import cv2
import numpy as np

# Ensure MediaPipe and local modules are in path (MediaPipe kept for backward compat if needed)
sys.path.append(os.getcwd())
sys.path.extend(site.getsitepackages())

import streamlit as st
from src.models.yolo_handler import YoloHandler
from src.utils.fall_logic import FallDetector

def main():
    st.set_page_config(page_title="Elderly Fall Detection Dashboard", layout="wide")
    
    st.title("🛡️ Patient Safety: AI Fall Detection Dashboard")
    st.markdown("---")
    
    # Sidebar: Configuration
    st.sidebar.header("System Settings")
    input_source = st.sidebar.radio("Input Source", ["Webcam", "Upload Image/Video"])
    
    # Unified Pipeline: MediaPipe is replaced by YOLOv11-Pose
    st.sidebar.success("Engine: YOLOv11-Pose (Active)")
    
    uploaded_file = None
    if input_source == "Upload Image/Video":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'])
    
    display_skeleton = st.sidebar.checkbox("Show Skeleton Overlay", value=True)
    display_bbox = st.sidebar.checkbox("Show Bounding Box", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Study: Comparative Analysis of YOLO and MediaPipe for Elderly Care.")
    
    if 'report' not in st.session_state:
        st.session_state['report'] = "No detection events recorded yet."
        
    st.sidebar.download_button(
        label="📄 Download Detection Report",
        data=st.session_state['report'],
        file_name="fall_detection_study_report.txt",
        mime="text/plain"
    )

    # Layout: Metrics and Video
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Real-Time Feed")
        frame_placeholder = st.empty()
        
    with col2:
        st.subheader("Safety Metrics")
        status_box = st.empty()
        velocity_metric = st.empty()
        angle_metric = st.empty()
        latency_metric = st.empty()
        probability_metric = st.empty()
        st.markdown("---")
        st.markdown("**System Audit Report**")
        audit_box = st.empty()
        
    # Initialize Core Engine (Unified YOLO-Pose)
    yolo = YoloHandler(model_path='yolo11s-pose.pt', conf=0.2)
    detector = FallDetector()
    
    # Start Video Stream (Simulated or Webcam)
    cap = None
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif uploaded_file is not None:
        # Handle uploaded image/video
        tfile = "temp_input" + os.path.splitext(uploaded_file.name)[1]
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
        
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            cap = cv2.VideoCapture(tfile)
        else:
            # It's an image
            frame = cv2.imread(tfile)
            if frame is not None:
                process_frame_logic(frame, yolo, detector, display_bbox, display_skeleton, frame_placeholder, status_box, velocity_metric, angle_metric, latency_metric, probability_metric, audit_box)
                st.info("Static image processed with High-Accuracy Mode.")
                return 

    stop_button = st.sidebar.button("Stop System")
    
    if cap is not None and cap.isOpened():
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            process_frame_logic(frame, yolo, detector, display_bbox, display_skeleton, frame_placeholder, status_box, velocity_metric, angle_metric, latency_metric, probability_metric, audit_box)
            time.sleep(0.01)
            
    if cap: cap.release()

def process_frame_logic(frame, yolo, detector, display_bbox, display_skeleton, frame_placeholder, status_box, velocity_metric, angle_metric, latency_metric, probability_metric, audit_box):
    start_ts = time.time()
    
    # --- Inference Phase (Unified YOLO-Pose) ---
    yolo_detections = yolo.process_frame(frame)
    
    # --- Keypoint Extraction (YOLO-Pose Mapping) ---
    neck, hips = None, None
    if yolo_detections and yolo_detections[0].get('keypoints') is not None:
        kpts = yolo_detections[0]['keypoints']
        # Map YOLO COCO Keypoints to Logic (5,6: shoulders | 11,12: hips)
        if len(kpts) >= 13:
            # Confidence check for shoulders
            if kpts[5][2] > 0.4 and kpts[6][2] > 0.4:
                neck = ((kpts[5][0] + kpts[6][0])/2, (kpts[5][1] + kpts[6][1])/2)
            # Confidence check for hips
            if kpts[11][2] > 0.4 and kpts[12][2] > 0.4:
                hips = ((kpts[11][0] + kpts[12][0])/2, (kpts[11][1] + kpts[12][1])/2)
    
    # --- Detection Logic ---
    is_fall, is_long_lie, debug = False, False, {"vy": 0, "angle": 0, "probability": 0}
    
    if yolo_detections:
        centroid_y = yolo_detections[0]['centroid'][1]
        bbox = yolo_detections[0]['bbox']
        is_fall, is_long_lie, debug = detector.update(centroid_y, neck, hips, bbox)
    
    # --- Visualization ---
    if display_bbox and yolo_detections:
        frame = yolo.draw_detections(frame, yolo_detections)
    if display_skeleton and yolo_detections:
        frame = yolo.draw_skeleton(frame, yolo_detections)
    
    # Update Dashboard
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    
    if is_fall:
        status_box.error("🚨 FALL DETECTED!")
    elif is_long_lie:
        status_box.warning("⚠️ CRITICAL: LONG LIE!")
    else:
        status_box.success("✅ STATUS: SAFE")
        
    velocity_metric.metric("Vertical Velocity", f"{debug['vy']:.2f} m/s")
    angle_metric.metric("Body Angle", f"{debug['angle']:.1f}°")
    probability_metric.metric("Fall Probability", f"{debug['probability']}%")
    latency_metric.metric("System Latency", f"{(time.time()-start_ts)*1000:.1f} ms")
    
    # Audit Report Breakdown
    indicators = debug.get('indicators', {})
    report_text = "\n".join([f"- **{k}**: {v}" for k, v in indicators.items()])
    audit_box.markdown(report_text)
    
    # Store formal report in session state
    if is_fall or is_long_lie:
        status_str = "FALL DETECTED" if is_fall else "CRITICAL LONG LIE"
        st.session_state['report'] = f"""
FALL DETECTION STUDY REPORT
===========================
Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
Overall Status: {status_str}
Confidence Score: {debug['probability']}%

DETECTION INDICATORS:
---------------------
- Vertical Velocity: {debug['vy']:.2f} m/s ({indicators.get('Velocity')})
- Skeletal Angle: {debug['angle']:.1f} degrees ({indicators.get('Posture')})
- Box AR: {debug['ar']:.2f} ({indicators.get('Box Shape')})

Note: This report is generated by the AI Fall Detection Prototype for thesis research.
        """

if __name__ == "__main__":
    main()
