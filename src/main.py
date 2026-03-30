import cv2
import time
from models.yolo_handler import YoloHandler
from models.mp_handler import MediaPipeHandler
from utils.fall_logic import FallDetector
from utils.data_loader import DatasetLoader

def run_pipeline(source=0):
    """
    Main execution loop integrating Pipeline A (YOLO) and Pipeline B (MediaPipe).
    """
    # Initialize Handlers
    yolo = YoloHandler()
    mp_pose = MediaPipeHandler()
    detector = FallDetector()
    loader = DatasetLoader(dataset_root='data/raw')
    
    cap = cv2.VideoCapture(source)
    
    print("Starting Fall Detection Console... Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.time()
        
        # --- PIPELINE A: YOLOv26 ---
        yolo_detections = yolo.process_frame(frame)
        
        # --- PIPELINE B: MediaPipe ---
        landmarks, h, w = mp_pose.process_frame(frame)
        neck, hips = mp_pose.get_key_points(landmarks, h, w)
        
        # --- FALL LOGIC INTEGRATION ---
        # For simplicity, we use the first person detected by YOLO to feed the detector
        if yolo_detections and neck and hips:
            main_person = yolo_detections[0]
            bbox = main_person['bbox']
            centroid_y = main_person['centroid'][1]
            
            is_fall, is_long_lie, debug = detector.update(centroid_y, neck, hips, bbox)
            
            # --- VISUALIZATION ---
            frame = yolo.draw_detections(frame, yolo_detections)
            frame = mp_pose.draw_skeleton(frame, landmarks)
            
            # Overlay Status
            status_text = "STATUS: SAFE"
            color = (0, 255, 0)
            if is_fall:
                status_text = "FALL DETECTED!"
                color = (0, 0, 255)
            if is_long_lie:
                status_text = "CRITICAL: LONG LIE!"
                color = (0, 0, 150)
                
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Draw Velocity/Angle info
            cv2.putText(frame, f"Vel: {debug['vy']:.2f} | Angle: {debug['angle']:.1f}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        latency = (time.time() - start_time) * 1000
        fps = 1 / (time.time() - start_time)
        
        cv2.putText(frame, f"Latency: {latency:.1f}ms | FPS: {fps:.1f}", (int(w-250), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('MSc Dissertation - Fall Detection Dashboard', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # In a real scenario, this would take CLI arguments for source or dataset mode
    run_pipeline(source=0) # Uses webcam by default
