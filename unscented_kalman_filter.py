import numpy as np


class UnscentedKalmanFilter:

    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        # state mean = [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # state covariance
        self.P = np.eye(4, 4) * 0.1

        # state dimension (number of states)
        self.L = len(self.x)

        # Determines spread of sigma points around the mean (usually a small positive number e.g. 1e-3)
        self.alpha = alpha

        # Incorporates prior knowledge of the distribution of the mean. For Gaussian distribution beta=2 is optimal
        self.beta = beta

        # Secondary scaling parameter (Usually set to 0 or 3-L)
        self.kappa = kappa

        self.Q = Q
        self.R = R

        self.is_initialized = False
        self.timestamp_epoch = 0.0

    def initialize(self, x: np.ndarray, timestamp_epoch: float):
        # initialize the mean and covariance

        # [x, y, vx, vy]
        self.x = x

        self.L = len(self.x)

        # state covariance
        self.P = np.eye(4, 4) * 0.1

        # Calculate lambda, the scaling parameter for the spread of sigma points
        # lambda = alpha^2 * (L + kappa) - L
        self.lam = self.alpha**2 * (self.L + self.kappa) - self.L

        # update the current timestamp
        self.timestamp_epoch = timestamp_epoch

        # compute the weight
        self._compute_weights()

        # update the status
        self.is_initialized = True

    def is_initialised(self) -> bool:
        """Get the initialisation status of the filter."""
        return self.is_initialized

    def _process_model(self, x, dt) -> np.ndarray:
        # Process model for constant velocity motion
        # State vector x: [x_position, y_position, x_velocity, y_velocity] (shape: 4x1)
        # dt: time step
        # Returns the predicted next state after dt seconds
        xout = np.zeros_like(x)

        # Update position using velocity and time step
        xout[0, 0] = x[0, 0] + x[2, 0] * dt  # x_position += x_velocity * dt
        xout[1, 0] = x[1, 0] + x[3, 0] * dt  # y_position += y_velocity * dt

        # Velocity remains constant (no acceleration modeled)
        xout[2, 0] = x[2, 0]  # x_velocity
        xout[3, 0] = x[3, 0]  # y_velocity

        return xout

    def _measurement_model(self, x) -> np.ndarray:
        # Measurement model for UKF
        # This function extracts the observed measurement from the state vector.
        # Only the position components [x_position, y_position] are observed.

        # Create a 2x1 measurement vector initialized to zeros
        xout = np.zeros((2, 1))

        xout[0, 0] = x[0, 0]  # Extract x_position from the state vector
        xout[1, 0] = x[1, 0]  # Extract y_position from the state vector

        return xout

    def predict(self, timestamp_epoch) -> np.ndarray:
        """
        UKF prediction step: propagates sigma points, computes predicted mean and covariance.

        Parameters
        ----------
        timestamp_epoch : float
            Current time in seconds (epoch). Used to compute time elapsed since last prediction.
        """

        if not self.is_initialized:
            raise RuntimeError("UKF must be initialized before predict().")

        # calculate time elapsed since last update (in seconds)
        dt = timestamp_epoch - self.timestamp_epoch

        # update the filter's timestamp to the current epoch
        self.timestamp_epoch = timestamp_epoch

        if dt <= 0:
            # skip propagation but still return current state
            return self.x

        # ------ UKF Prediction Step ------------

        # generate sigmal points
        self.generate_sigma_points()

        # predict sigma point through the process model
        self.predict_sigma_points(dt)

        # compute predicted state mean
        pred_X = np.zeros_like(self.x)
        for i in range(self.pred_sigma_points.shape[1]):
            pred_X += self.Wm[i] * self.pred_sigma_points[:, i].reshape(-1, 1)

        # compute predicted state covariance
        pred_P = self.Q.copy().astype(np.float64)
        for i in range(self.pred_sigma_points.shape[1]):
            diff = self.pred_sigma_points[:, i].reshape(-1, 1) - pred_X
            pred_P += self.Wc[i] * (diff @ diff.transpose())

        self.x = pred_X  # predicted state mean
        self.P = pred_P  # predicted state covariance

        return self.x

    def update(self, measurement: np.ndarray) -> np.ndarray:
        # Set measurement dimension (number of observed variables)
        n_z = len(measurement)

        # Transform predicted sigma points into measurement space using the measurement model
        Z = np.zeros((n_z, 2 * self.L + 1))
        for i in range(self.pred_sigma_points.shape[1]):
            x = self.pred_sigma_points[:, i].reshape(-1, 1)
            Z[:, i] = self._measurement_model(x).flatten()

        # ------ UKF Update Step ------------

        # Compute predicted measurement mean (weighted sum of transformed sigma points)
        pred_Z = np.zeros_like(measurement)
        for i in range(Z.shape[1]):
            pred_Z += self.Wm[i] * Z[:, i].reshape(-1, 1)

        # Compute residual: difference between actual and predicted measurement
        y = measurement - pred_Z

        # Compute innovation covariance S (measurement uncertainty)
        S = self.R.copy().astype(np.float64)
        for i in range(Z.shape[1]):
            diff = Z[:, i].reshape(-1, 1) - pred_Z
            S += self.Wc[i] * (diff @ diff.transpose())

        # Compute cross covariance Pxz between state and measurement
        Pxz = np.zeros((self.L, n_z))
        for i in range(Z.shape[1]):
            dx = self.pred_sigma_points[:, i].reshape(-1, 1) - self.x
            dz = Z[:, i].reshape(-1, 1) - pred_Z
            Pxz += self.Wc[i] * (dx @ dz.transpose())

        # Compute Kalman Gain (how much to correct the prediction)
        K = Pxz @ np.linalg.inv(S)

        # Update state mean with measurement
        self.x = self.x + (K @ y)

        # Update state covariance
        self.P = self.P - (K @ S @ K.transpose())

        return self.x

    def _compute_weights(self):
        # Compute the scaling factor for the weights
        # c = 1 / [2 * (L + lambda)]
        c = 1.0 / (2.0 * (self.L + self.lam))

        # Initialize mean weights for all sigma points
        self.Wm = np.full(2 * self.L + 1, c)

        # Initialize covariance weights
        self.Wc = self.Wm.copy()

        # Set the first mean weight
        # Wm[0] = lambda / (L + lambda)
        self.Wm[0] = self.lam / (self.L + self.lam)

        # Set the first covariance weight
        # Wc[0] = lambda / (L + lambda) + (1 - alpha^2 + beta)
        self.Wc[0] = self.lam / (self.L + self.lam) + (1.0 - self.alpha**2 + self.beta)

    def generate_sigma_points(self):

        # Compute the Cholesky decomposition of the state covariance matrix P
        A = np.linalg.cholesky(self.P)

        # Gamma is the scaling factor for the sigma points
        gamma = np.sqrt(self.L + self.lam)

        # Initialize the sigma points array
        # Shape: (state dimension, 2L + 1)
        self.sigma_points = np.zeros((self.L, 2 * self.L + 1))

        # The first sigma point is the mean state vector
        self.sigma_points[:, 0] = self.x.flatten()

        # generate the remaining sigma points
        for i in range(self.L):
            # Sigma point in the positive direction
            self.sigma_points[:, i + 1] = self.x.flatten() + (gamma * A[:, i])
            # Sigma point in the negative direction
            self.sigma_points[:, i + 1 + self.L] = self.x.flatten() - (gamma * A[:, i])

    def predict_sigma_points(self, dt):
        pred_sigma_points = np.zeros_like(self.sigma_points)
        for i in range(self.sigma_points.shape[1]):
            x = self.sigma_points[:, i].reshape(-1, 1)
            # Apply process model to propagate sigma point
            pred_sigma_points[:, i] = self._process_model(x, dt).flatten()

        self.pred_sigma_points = pred_sigma_points
