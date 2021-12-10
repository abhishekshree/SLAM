import numpy as np


class Landmark(object):
    def __init__(self, identity, mu: np.ndarray, sig: np.ndarray):
        self.identity = identity
        self.mu = mu
        self.sig = sig

    def update(self, mu, sig):
        self.mu = mu
        self.sig = sig
