import numpy as np


class KalmanFilter:

    def __init__(self, Q: np.ndarray, R: np.ndarray):
        # state mean = [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        # state covariance
        self.P = np.eye(4, 4)

        self.Q = Q
        self.R = R

        self.is_initialized = False
        self.timestamp_epoch = 0.0

    def initialize(self, x: np.ndarray, timestamp_epoch: float):
        # initialize the mean and covariance

        # [x, y, vx, vy]
        self.x = x

        # state covariance
        self.P = np.eye(4, 4)

        # update the current timestamp
        self.timestamp_epoch = timestamp_epoch

        # update the status
        self.is_initialized = True

    def is_initialised(self) -> bool:
        """Get the initialisation status of the filter."""
        return self.is_initialized

    def predict(self, timestamp_epoch) -> np.ndarray:
        # time elapsed in seconds
        dt = timestamp_epoch - self.timestamp_epoch

        # Update timestamp
        self.timestamp_epoch = timestamp_epoch

        if dt <= 0:
            # skip propagation but still return current state
            return self.x

        # State Transition Matrix F
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Predict the new state mean
        self.x = F @ self.x

        # Predict the new state covariance
        self.P = (F @ self.P @ F.transpose()) + self.Q

        return self.x

    def update(self, measurement: np.ndarray) -> np.ndarray:
        # Measurment Matrix H (2 x 4)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Residual (difference between the actual and the predicted measurement)
        y = measurement - H @ self.x

        # measures the uncertainty in the measurement update
        S = (H @ self.P @ H.transpose()) + self.R

        # Kalman Gain (determines how much the prediction should be adjusted by the innovation)
        K = self.P @ H.transpose() @ np.linalg.inv(S)

        # Update the state mean
        self.x = self.x + (K @ y)

        # Update the state covarinace
        I = np.eye(4)
        self.P = (I - (K @ H)) @ self.P

        return self.x
