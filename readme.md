# Kalman Filter Bicycle Tracking

This project demonstrates tracking a bicycle in a video using object detection (YOLO) and state estimation (Kalman Filter and Unscented Kalman Filter).

## Features

- Detects bicycles in video frames using a YOLO model.
- Tracks the detected bicycle using Kalman Filter or Unscented Kalman Filter.
- Visualizes tracking results frame-by-frame.

### Files

- `main.py`: Main script for running detection and tracking on a video.
- `kalman_filter.py`: Standard Kalman Filter implementation.
- `unscented_kalman_filter.py`: Unscented Kalman Filter implementation.
- `yolo11s.pt`: YOLO model weights for bicycle detection.
- `VIRAT_S_010204_09_001285_001336.mp4`: Example video for tracking.

### Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

Install dependencies:

```bash
pip install opencv-python numpy ultralytics
```

### Usage

Run the main script:

```bash
python main.py
```

You can change the video path or model weights in `main.py` as needed.

### Reference

- [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter)
- [Unscented Kalman Filter](https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf)
