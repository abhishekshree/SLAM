#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, mu, cov):
        # Matrix n x n that describes how the state evolves from t-1 to t without controls or noise
        self.A = A
        # Matrix n x l that describes how the control changes the state from t-1 to t.
        self.B = B
        # Matrix k x n that describes how to map the state x_t to an observation z_t.
        self.C = C
        # Measurement Noise
        self.Q = Q
        # Motion Noise
        self.R = R
        
        self.mu = mu
        self.cov = cov

        self.count = 0
        

    def iterate(self, u, z):
        mu_bel = self.A @ self.mu + self.B @ u
        cov_bel = self.A @ self.cov @ self.A.T + self.R
        K = cov_bel @ self.C.T @ LA.inv(self.C @ cov_bel @ self.C.T + self.Q)

        self.mu = mu_bel + K @ (z - self.C @ mu_bel)
        self.cov = (np.eye(self.cov.shape[0]) - K @ self.C) @ cov_bel

        self.count += 1

        return self.mu, self.cov
