import cv2
import numpy as np

class StressTester:
    """
    Environmental Stress Testing Suite.
    Simulates Low Light and Occlusion scenarios to measure model robustness.
    """
    
    @staticmethod
    def simulate_low_light(frame, alpha=0.5, beta=-30):
        """
        Reduces brightness and contrast to simulate poor lighting.
        g(x,y) = alpha * f(x,y) + beta
        """
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    @staticmethod
    def simulate_occlusion(frame, section='bottom'):
        """
        Occludes parts of the frame to simulate furniture or obstacles.
        """
        h, w, _ = frame.shape
        if section == 'bottom':
            # Block the bottom 40% (legs/floor)
            frame[int(h*0.6):, :, :] = 0
        elif section == 'center':
            # Block a central pillar
            frame[:, int(w*0.4):int(w*0.6), :] = 0
            
        return frame

    def run_stress_test(self, frame, mode='light'):
        if mode == 'light':
            return self.simulate_low_light(frame)
        elif mode == 'occlusion':
            return self.simulate_occlusion(frame)
        return frame
