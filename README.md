# Automating Patient Safety: Fall Detection Comparative Analysis (YOLO-Pose Edition)

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://falldetectionpipeline.streamlit.app/)

This repository contains the full technical implementation for the MSc Dissertation: **"Automating Patient Safety: A Comparative Analysis of YOLOv26 and MediaPipe for Real-Time Fall Detection in Elderly Care Environments."**

> [!NOTE]
> **Update:** This project has been optimized to run on **Python 3.14+** by migrating from MediaPipe to a unified **YOLOv11s-Pose** engine, ensuring high-fidelity skeletal tracking and detection stability.

## 🚀 Overview
The system implements a robust fall detection dashboard using **YOLOv11s-Pose**. It utilizes a multi-modal mathematical heuristic based on vertical velocity, posture angles, and aspect ratios to identify falls in both video streams and static images.

## 🛠️ Key Features
- **Unified YOLO-Pose Pipeline**: Real-time person detection and 17-point skeletal tracking in a single pass.
- **Fall Probability Scoring**: A weighted 0-100% confidence metric based on heuristic overlap.
- **Research Audit Reports**: Automated generation of detection summaries (Velocity, Angle, Bbox AR) for study documentation.
- **Velocity Smoothing**: Integrated EMA filter to eliminate sensor jitter and false positives.
- **Privacy-by-Design**: Options to toggle Bounding Boxes and Skeletons for GDPR compliance.

## 📂 Project Structure
- `app.py`: Main Streamlit Dashboard.
- `src/models/yolo_handler.py`: Unified YOLO-Pose inference engine.
- `src/utils/fall_logic.py`: Core heuristic state machine and probability calculator.
- `src/models/mp_handler.py`: Legacy MediaPipe handler (Deprecated for Python 3.14).

## 🖥️ Local Installation

To run this project on your local Windows/Linux machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/azeem-aslam-ch/falldetectionpipeline.git
   cd falldetectionpipeline
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard:**
   ```bash
   python -m streamlit run app.py
   ```

## ☁️ Deployment on Streamlit Cloud

You can deploy this dashboard for free on the web:

1. **Push your code** to your GitHub repository (already done if you're reading this).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Click **"New App"** and select:
   - **Repository:** `azeem-aslam-ch/falldetectionpipeline`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**

## 📊 Methodology
The system employs a multi-condition trigger for falls:
1. **Vertical Velocity ($V_y$):** Detection of sudden downward acceleration (Smoothed).
2. **Posture Angle ($\Theta$):** Body inclination relative to the horizontal plane via YOLO-Pose keypoints.
3. **Bounding Box AR:** Width-to-Height ratio analysis.
4. **Probability Score:** Weighted sum of indicators ($V_y \times 0.4 + \Theta \times 0.4 + AR \times 0.2$).

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
