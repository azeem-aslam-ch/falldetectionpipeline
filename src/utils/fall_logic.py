import numpy as np
import time

class FallDetector:
    """
    Core Heuristic Engine for Fall Detection.
    Implements a multi-modal state machine based on velocity, posture, and aspect ratio.
    """
    def __init__(self, velocity_threshold=2.5, angle_threshold=30, ar_threshold=1.2, long_lie_threshold=5.0):
        self.v_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
        self.ar_threshold = ar_threshold
        self.long_lie_threshold = long_lie_threshold
        
        # State tracking
        self.prev_y = None
        self.prev_t = None
        self.fall_start_time = None
        self.is_falling_candidate = False
        self.is_long_lie = False

    def calculate_velocity(self, y_curr, t_curr):
        """Calculates smoothed vertical velocity using Exponential Moving Average."""
        # 1. Initialization
        if self.prev_y is None or self.prev_t is None:
            self.prev_y, self.prev_t = y_curr, t_curr
            return 0.0
        
        # 2. Timing
        dt = t_curr - self.prev_t
        if dt <= 0.001: return 0.0 # Prevent division by near-zero
        
        # 3. Smoothing (EMA)
        # Smoothing factor: 0.3 (lower is smoother, higher is more responsive)
        y_smoothed = 0.3 * y_curr + 0.7 * self.prev_y
        
        # 4. Calculation (Scaled for pixel density)
        # Assuming person height is ~1.7m, the normalization helps stabilize across resolutions
        vy = (y_smoothed - self.prev_y) / dt
        
        # Scale down for normalization (pixels to estimated m/s)
        vy_scaled = vy / 100.0 
        
        self.prev_y, self.prev_t = y_smoothed, t_curr
        return vy_scaled

    def calculate_posture_angle(self, neck, hips):
        """
        Calculates body inclination angle relative to the horizontal plane.
        Expected inputs are (x, y) tuples.
        """
        dx = hips[0] - neck[0]
        dy = hips[1] - neck[1]
        
        # Angle with vertical axis
        angle_rad = np.arctan2(abs(dx), abs(dy))
        angle_deg = np.degrees(angle_rad)
        
        # Return angle relative to horizontal (90 = vertical, 0 = horizontal)
        return 90 - angle_deg

    def calculate_aspect_ratio(self, bbox):
        """Calculates Width/Height ratio of the bounding box."""
        x, y, w, h = bbox
        if h == 0: return 0.0
        return w / h

    def update(self, y_curr, neck_curr, hips_curr, bbox_curr):
        """
        Processes a single frame and returns the detection state.
        Returns: (is_fall, is_long_lie, debug_info)
        """
        t_curr = time.time()
        
        # 1. Feature Extraction
        vy = self.calculate_velocity(y_curr, t_curr)
        ar = self.calculate_aspect_ratio(bbox_curr)
        
        # Posture angle calculation (requires landmarks)
        angle = 90.0 # Default to vertical if no landmarks
        if neck_curr and hips_curr:
            angle = self.calculate_posture_angle(neck_curr, hips_curr)
        
        fall_detected = False
        
        # Adaptive Logic: Use skeletal angle if available, otherwise rely more on AR and Velocity
        if neck_curr and hips_curr:
            # Combined Logic: Dynamic Fall (High Velocity) OR Static Fall (Lying Down)
            is_dynamic = (vy > self.v_threshold or self.is_falling_candidate) and (angle < self.angle_threshold)
            is_static_lying = (angle < self.angle_threshold) and (ar > self.ar_threshold)
            condition = is_dynamic or is_static_lying
        else:
            # YOLO-Only Fallback Logic
            is_dynamic = (vy > self.v_threshold or self.is_falling_candidate) and (ar > self.ar_threshold * 1.5)
            is_static_lying = (ar > self.ar_threshold * 2.0) # High width/height ratio
            condition = is_dynamic or is_static_lying

        # 2. Logic Trigger
        # Condition A: Sudden downward velocity spike
        # Condition B: Horizontal posture angle (if available)
        # Condition C: Horizontal bounding box AR
        if condition:
            if not self.is_falling_candidate:
                self.fall_start_time = t_curr
                self.is_falling_candidate = True
            
            # 3. Long Lie Verification
            if t_curr - self.fall_start_time > self.long_lie_threshold:
                self.is_long_lie = True
            
            fall_detected = True
        else:
            # Reset if conditions not met
            self.is_falling_candidate = False
            self.is_long_lie = False
            self.fall_start_time = None

        # --- Probability Calculation ---
        # Heuristics:
        # 1. Velocity (Up to 40%): How fast is the downward movement?
        # 2. Angle (Up to 40%): How horizontal is the skeletal posture?
        # 3. Aspect Ratio (Up to 20%): How wide is the bounding box?
        
        v_prob = min(100, (abs(vy) / self.v_threshold) * 100) * 0.4
        a_prob = min(100, ((90 - angle) / (90 - self.angle_threshold)) * 100) * 0.4
        ar_prob = min(100, (ar / (self.ar_threshold * 2)) * 100) * 0.2
        
        prob_score = int(v_prob + a_prob + ar_prob)
        if condition: prob_score = max(prob_score, 85) # Force high prob if condition met
        if not y_curr: prob_score = 0 # No person, no prob

        debug_info = {
            "vy": vy,
            "angle": angle,
            "ar": ar,
            "duration": t_curr - self.fall_start_time if self.fall_start_time else 0,
            "probability": prob_score,
            "indicators": {
                "Velocity": "CRITICAL" if abs(vy) > self.v_threshold else "Normal",
                "Posture": "HORIZONTAL" if angle < self.angle_threshold else "Vertical",
                "Box Shape": "WIDE (Lying)" if ar > self.ar_threshold else "Narrow (Standing)"
            }
        }
        
        return fall_detected, self.is_long_lie, debug_info
