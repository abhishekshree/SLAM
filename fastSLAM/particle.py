from landmark import Landmark
import numpy as np


class Particle(object):
    def __init__(self, state) -> None:
        self.weight = 0.5
        self.state = state
        self.__landmarks = dict()

    def getWeight(self) -> float:
        return self.weight

    def getState(self) -> list:
        return self.state

    def getLandmarksMap(self) -> dict:
        return self.__landmarks

    def setWeight(self, weight: float) -> None:
        self.weight = weight

    def setState(self, state: list) -> None:
        self.state = state

    def setLandmarksMap(self, landmarks: dict) -> None:
        self.__landmarks = landmarks

    def getLandmarkCount(self) -> int:
        return len(self.__landmarks)

    def getLandmark(self, index) -> Landmark:
        return self.__landmarks[index]

    def setLandmark(self, index, landmark: Landmark) -> None:
        self.__landmarks[index] = landmark

    def updateWeight(self, Q: np.ndarray, z: np.ndarray) -> None:
        self.weight = (
            2
            * np.pi
            * np.linalg.det(Q) ** -0.5
            * np.exp(
                -0.5
                * np.dot(
                    np.dot(np.transpose(z - self.state), np.linalg.inv(Q)),
                    z - self.state,
                )
            )
        )
