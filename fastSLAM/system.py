import numpy as np
from landmark import Landmark

from particle import Particle


class System:
    def __init__(self, particle_count: int, G, h, Q) -> None:
        self.particle_count = particle_count
        self.particle_system = [Particle for _ in range(particle_count)]
        self.G = G
        self.h = h
        self.Q = Q
        self.__default_importance_weight = 1.0 / particle_count

    def get_particle_count(self) -> int:
        return self.particle_count

    def get_particle_system(self) -> np.ndarray:
        return self.particle_system

    def get_G(self) -> float:
        return self.G

    # TODO: implement
    def motion_model() -> None:
        pass

    def jacobian_h(self, x, j) -> np.ndarray:
        pass

    def EKFUpdate(self, z_t, h, Q_t) -> None:
        pass

    def update_system(self, z_t, u_t, c_t) -> None:
        j = c_t  # Feature
        for p in self.particle_system:
            p.setState(self.motion_model(p.getState(), u_t, j))

            if j not in p.getLandmarksMap():
                mu = np.random.normal(size=2)
                H = self.jacobian_h(p.getState(), j)
                sig = np.linalg.inv(H) * self.Q * np.linalg.inv(H).T
                l = Landmark(j, mu, sig)
                p.setLandmark(j, l)
                p.setWeight(self.__default_importance_weight)

            else:
                mu, sig = self.EKFUpdate(z_t, self.h, self.Q)
                p.setLandmark(j, Landmark(j, mu, sig))
                p.updateWeight(Q=self.Q, z=z_t)

                for ap in self.particle_system:
                    ap.setLandmark(j, Landmark(j, mu, sig))

    def resample_system(self, threshold) -> None:
        for p in self.particle_system:
            if p.getWeight() < threshold:
                self.particle_system.remove(p)

    def mean(self) -> float:
        pass
