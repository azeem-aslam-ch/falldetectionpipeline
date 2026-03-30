import os
import cv2

class DatasetLoader:
    """
    Handler for UR Fall Detection (URFD) and Multiple Camera Fall Dataset (MCFD).
    Provides a standardized interface for frame extraction and annotation loading.
    """
    def __init__(self, dataset_root):
        self.root = dataset_root
        self.urfd_path = os.path.join(dataset_root, 'URFD')
        self.mcfd_path = os.path.join(dataset_root, 'MCFD')

    def get_video_paths(self, dataset_name='urfd'):
        """Returns a list of video file paths for the specified dataset."""
        target_path = self.urfd_path if dataset_name.lower() == 'urfd' else self.mcfd_path
        if not os.path.exists(target_path):
            print(f"Warning: Dataset path {target_path} not found.")
            return []
            
        videos = []
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    videos.append(os.path.join(root, file))
        return videos

    def load_annotations(self, video_path):
        """
        Stub for loading dataset-specific annotations (CSV/TXT).
        In a real dissertation, this would parse frame-level fall labels.
        """
        # Placeholder for annotation parsing logic
        return None

    def stream_source(self, source=0):
        """
        Provides a generator for frames from a given source (webcam or video).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            
        cap.release()
