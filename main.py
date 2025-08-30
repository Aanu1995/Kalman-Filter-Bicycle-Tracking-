import os
import time

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from kalman_filter import KalmanFilter
from unscented_kalman_filter import UnscentedKalmanFilter


def predict(model: YOLO, frame: np.ndarray):

    classes = [1]  # Only detect bicycle in the frame
    results = model.predict(frame, conf=0.5, classes=classes)

    box = None  # Bounding Box of bicycle detected
    anotated_frame = frame

    if len(results) > 0:
        result = results[0]

        anotated_frame = result.plot(conf=False)
        boxes = list(result.boxes.xywh.cpu())

        if len(boxes) > 0:
            box = boxes[0]

        return anotated_frame, box

    return anotated_frame, box


def main(video_path: str):
    # use YOLO to detect the bicycle
    model = YOLO("yolo11s.pt")

    # Open the video file
    cap = cv.VideoCapture(video_path)

    timestamp_epoch = 0.0
    Q = np.diag([1e-2, 1e-2, 1e-2, 1e-2])  # More uncertainty in velocity
    R = np.diag([4.0, 4.0])  # Measurement noise (adjust based on sensor)

    # kf = KalmanFilter(Q, R)
    kf = UnscentedKalmanFilter(alpha=0.1, beta=2.0, kappa=0, Q=Q, R=R)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Get timestamp of the current time in seconds since the Epoch
        timestamp_epoch = time.time()

        annotated_frame, box = predict(model, frame)

        # if kalman filter state is already initialized
        if kf.is_initialised():
            if box is not None:

                predicted_state_mean = kf.predict(timestamp_epoch)
                x, y, _, _ = predicted_state_mean
                # Show the predicted position of the bicycle on the image
                cv.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)

                cx, cy, _, _ = box
                measurement = np.array([cx, cy], dtype=np.float64).reshape(2, 1)
                updated_state_mean = kf.update(measurement)
                x, y, _, _ = updated_state_mean
                # Show the updated position of the bicycle on the image
                cv.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            else:
                # Predict the position only without update
                # because the measurement from bounding box is not available
                predicted_state_mean = kf.predict(timestamp_epoch)
                x, y, _, _ = predicted_state_mean

                # Show the predicted position of the bicycle on the image
                cv.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        elif box is not None:
            # initialize the state mean if not already initialized
            cx, cy, _, _ = box
            x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64).reshape(4, 1)
            kf.initialize(x, timestamp_epoch)

        cv.imshow("Bicycle Tracking", annotated_frame)
        if cv.waitKey(1) == ord("q"):  # Break the loop if 'q' is pressed
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(root_dir, "VIRAT_S_010204_09_001285_001336.mp4")
    main(video_path)
