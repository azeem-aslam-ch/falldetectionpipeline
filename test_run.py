import cv2
import os
import time
from src.models.yolo_handler import YoloHandler
from src.models.mp_handler import MediaPipeHandler
from src.utils.fall_logic import FallDetector

def run_verified_demo(image_path, output_path):
    print(f"Initializing Fall Detection Engine...")
    yolo = YoloHandler(model_path='yolo11s.pt')
    mp_pose = MediaPipeHandler()
    detector = FallDetector()
    
    print(f"Loading test scene: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return
    
    h, w, _ = frame.shape
    
    # 1. Processing
    yolo_detections = yolo.process_frame(frame)
    landmarks, _, _ = mp_pose.process_frame(frame)
    neck, hips = mp_pose.get_key_points(landmarks, h, w)
    
    # 2. Logic
    is_fall, is_long_lie, debug = False, False, {}
    if yolo_detections and neck and hips:
        centroid_y = yolo_detections[0]['centroid'][1]
        bbox = yolo_detections[0]['bbox']
        is_fall, is_long_lie, debug = detector.update(centroid_y, neck, hips, bbox)
    
    # 3. Annotation
    if yolo_detections:
        frame = yolo.draw_detections(frame, yolo_detections)
    if landmarks:
        frame = mp_pose.draw_skeleton(frame, landmarks)
        
    status_text = "STATUS: SAFE"
    color = (0, 255, 0)
    if is_fall:
        status_text = "FALL DETECTED!"
        color = (0, 0, 255)
    
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"Velocity: {debug.get('vy', 0):.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, frame)
    print(f"Demo Successful! Result saved to {output_path}")
    print(f"Detection Status: {status_text}")

if __name__ == "__main__":
    img_path = r"C:\Users\chaze\.gemini\antigravity\brain\87f14f73-faad-467b-8425-249458535c40\fall_detection_demo_scene_1774115749792.png"
    out_path = "demo_result_verified.png"
    run_verified_demo(img_path, out_path)
