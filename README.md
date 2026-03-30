# Automating Patient Safety: Fall Detection Comparative Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the full technical implementation for the MSc Dissertation: **"Automating Patient Safety: A Comparative Analysis of YOLOv26 and MediaPipe for Real-Time Fall Detection in Elderly Care Environments."**

## 🚀 Overview
The system implements a dual-pipeline approach to fall detection, comparing the efficiency of frame-level object detection (YOLOv26) against landmark-based pose estimation (MediaPipe). It utilizes a custom mathematical heuristic based on vertical velocity, posture angles, and aspect ratios to identify falls while minimizing false positives.

## 🛠️ Key Features
- **Dual Inference Pipelines**: Simultaneous execution of YOLO and MediaPipe for benchmarking.
- **Privacy-by-Design**: Skeleton-only processing mode for GDPR compliance.
- **Heuristic Engine**: Velocity spike detection + Posture state machine.
- **Benchmarking Suite**: Automated calculation of Precision, Recall, F1, mAP, FPS, and Latency.
- **Edge Simulation**: Optimized for CPU-only inference.

## 📂 Project Structure
Refer to `folder_structure.txt` for the full directory map. Key directories:
- `src/models/`: Pipeline implementations.
- `src/utils/`: Core fall detection logic and dataset handlers.
- `src/eval/`: Benchmarking and stress-test scripts.
- `docs/Chapters/`: Full MSc Dissertation text.

## 📥 Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download URFD/MCFD datasets and place in `data/raw/`.

## 🖥️ Usage
To run the real-time demo:
```bash
streamlit run app.py
```

To run the evaluation benchmarks:
```bash
python src/eval/benchmarker.py --dataset urfd
```

## 📊 Methodology
The system employs a multi-condition trigger for falls:
1. **Vertical Velocity ($V_y$):** Detection of sudden downward acceleration.
2. **Posture Angle ($\Theta$):** Body inclination relative to the horizontal plane.
3. **Bounding Box AR:** Width-to-Height ratio analysis.
4. **Long Lie Logic:** Temporal thresholding to avoid triggers from sitting/crouching.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
