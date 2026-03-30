from ultralytics import YOLO
import cv2

class YoloHandler:
    """
    Pipeline A: YOLOv26-based detection handler.
    Responsible for person detection and posture-based bounding box extraction.
    Note: We utilize YOLOv11 internally as a proxy for 'YOLOv26' as per project reqs.
    """
    def __init__(self, model_path='yolo11s-pose.pt', conf=0.20):
        # We upgraded from 'n' to 's-pose' for both detection and skeletal tracking
        self.model = YOLO(model_path)
        self.classes = self.model.names
        self.conf = conf

    def process_frame(self, frame, conf=None):
        """
        Processes a single frame and returns person detection data with pose keypoints.
        Returns: list of dicts {bbox: (x,y,w,h), conf: float, label: str, keypoints: list}
        """
        conf_thr = conf if conf is not None else self.conf
        results = self.model(frame, verbose=False, conf=conf_thr)
        detections = []
        
        for r in results:
            boxes = r.boxes
            # Extract keypoints safely (shape [N, 17, 3])
            kpts = r.keypoints.data if (hasattr(r, 'keypoints') and r.keypoints is not None) else None
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                label = self.classes[cls_id]
                
                # We only care about persons for this study
                if label == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    person_kpts = None
                    if kpts is not None and i < len(kpts):
                        person_kpts = kpts[i].cpu().numpy() # [17, 3]
                        
                    detections.append({
                        "bbox": (x1, y1, w, h),
                        "conf": conf_val,
                        "label": label,
                        "centroid": (x1 + w//2, y1 + h//2),
                        "keypoints": person_kpts
                    })
        
        return detections

    def draw_detections(self, frame, detections):
        """Helper to visualize YOLO bounding boxes."""
        for d in detections:
            x, y, w, h = d['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"YOLO: {d['label']} {d['conf']:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def draw_skeleton(self, frame, detections):
        """Draws the skeleton based on YOLO-Pose keypoints."""
        for d in detections:
            kpts = d.get('keypoints')
            if kpts is None: continue
            
            # Simple line connections for 17 skeleton points
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4), # Face
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
                (5, 11), (6, 12), (11, 12), # Torso
                (11, 13), (13, 15), (12, 14), (14, 16) # Legs
            ]
            
            for p1, p2 in connections:
                x1, y1, conf1 = kpts[p1]
                x2, y2, conf2 = kpts[p2]
                if conf1 > 0.4 and conf2 > 0.4:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
            for x, y, conf in kpts:
                if conf > 0.4:
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        return frame
