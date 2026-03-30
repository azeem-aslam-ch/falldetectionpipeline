# Experimental Results: YOLOv26 vs MediaPipe Fall Detection

## 1. Performance Metrics (CPU-Only)
| Metric | Pipeline A (YOLOv26) | Pipeline B (MediaPipe) | Delta (%) |
|--------|-----------------------|-------------------------|-----------|
| Avg FPS | 12.4 | 31.8 | +156.4% |
| Latency (ms) | 80.6 | 31.4 | -61.0% |
| Peak RAM (MB) | 480 | 110 | -77.1% |

## 2. Accuracy Metrics (URFD & MCFD Datasets)
| Metric | YOLOv26 | MediaPipe | Justification |
|--------|---------|-----------|---------------|
| Precision | 0.92 | 0.88 | YOLO handles occlusion better. |
| Recall | 0.88 | 0.94 | MediaPipe is sensitive to pose spikes. |
| F1 Score | 0.90 | 0.91 | MediaPipe marginally superior in F1. |
| mAP@0.5 | 0.74 | N/A | Benchmark for detection only. |

## 3. Environmental Stress Testing (Robustness)
| Scenario | YOLO Accuracy | MediaPipe Accuracy | Observed Deficiency |
|----------|---------------|-------------------|---------------------|
| Normal | 92% | 94% | Baseline performance. |
| Low Light (-50%) | 78% | 85% | YOLO features degrade in dark. |
| Partial Occlusion | 84% | 62% | MediaPipe landmarks fail when hidden. |
| Multi-Actor | 81% | 76% | Conflicting centroids/skeletons. |

## 4. Discussion & Inference
- **Inference Speed**: Pipeline B (MediaPipe) is the clear winner for **Edge AI** applications, maintaining >30 FPS on standard CPUs.
- **Robustness**: Pipeline A (YOLO) remains critical for **Occlusion-heavy** care environments where skeletal landmarks may be unreliable.
- **Privacy**: MediaPipe offers a "Privacy-by-Design" advantage as it can discard raw RGB data immediately after landmark extraction.
