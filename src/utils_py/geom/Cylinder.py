from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .Shape import Shape

@dataclass
class Cylinder(Shape):
    radius: float = 0
    length: float = 0
    axis: np.array = field(default_factory=np.array)

    def get_volume(self) -> float:
        return np.pi * self.radius**2 * self.length

    def get_surface(self) -> float:
        return 2 * np.pi * self.radius * (self.radius + self.length)

    def check_point(self, point) -> bool:
        d = np.linalg.norm(np.cross(point - self.center, self.axis))
        l = np.abs(np.dot(self.axis, point - self.center))

        return (d < self.radius) and (l < self.length / 2)

    def generate_point(self) -> np.array:
        r = self.radius * np.sqrt(np.random.uniform(0, 1))
        # r = np.random.uniform(0, self.radius)
        phi = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-self.length / 2, self.length / 2)

        point = np.array([r * np.cos(phi), r * np.sin(phi), z])

        rot = Rotation.from_euler('ZY', [np.arctan(self.axis[1] / self.axis[0]), np.arccos(self.axis[2])])
        rot_point = rot.apply(point)

        return self.center + rot_point
