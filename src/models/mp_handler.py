import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.solutions import pose as mp_pose
    from mediapipe.solutions import drawing_utils as mp_drawing
    MP_AVAILABLE = True
except Exception as e:
    MP_AVAILABLE = False
    print(f"Warning: MediaPipe failed to load: {e}")

class MediaPipeHandler:
    """
    Pipeline B: MediaPipe Pose estimation handler.
    Extracts 33 body landmarks for skeletal analysis and privacy-preserving logic.
    """
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        if MP_AVAILABLE:
            self.mp_pose = mp_pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp_drawing
        else:
            self.pose = None

    def process_frame(self, frame):
        """
        Extracts landmarks from the frame.
        Returns: (landmarks, image_height, image_width)
        """
        if not self.pose:
            return None, 0, 0
            
        h, w, c = frame.shape
        # Convert to RGB as MediaPipe requires it
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        landmarks_data = None
        if results.pose_landmarks:
            landmarks_data = results.pose_landmarks
            
        return landmarks_data, h, w

    def get_key_points(self, landmarks, h, w):
        """
        Extracts specific joint coordinates for the fall logic.
        Focuses on Neck (Mid-Shoulder) and Hip Midpoint.
        """
        if not self.pose or not landmarks: return None, None
        
        # Landmarks indices:
        # Left Shoulder: 11, Right Shoulder: 12
        # Left Hip: 23, Right Hip: 24
        
        ls = landmarks.landmark[11]
        rs = landmarks.landmark[12]
        lh = landmarks.landmark[23]
        rh = landmarks.landmark[24]
        
        # Calculate Neck (midpoint between shoulders)
        neck_x = (ls.x + rs.x) / 2 * w
        neck_y = (ls.y + rs.y) / 2 * h
        
        # Calculate Hip Midpoint
        hip_x = (lh.x + rh.x) / 2 * w
        hip_y = (lh.y + rh.y) / 2 * h
        
        return (neck_x, neck_y), (hip_x, hip_y)

    def draw_skeleton(self, frame, landmarks):
        """Helper to visualize the MediaPipe skeleton."""
        if self.pose and landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
