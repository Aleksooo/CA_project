from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .Shape import Shape
from .Cylinder import Cylinder

@dataclass
class CylinderSegment(Shape):
    cylinder: Cylinder = field(default_factory=Cylinder)

    segm_radius: float = 0
    norm: np.array = field(default_factory=np.array) # should be perpendicular to axis

    def get_volume(self) -> float:
        alpha = 2 * np.arccos(self.segm_radius / self.cylinder.radius)

        return self.cylinder.radius**2 * (alpha - np.sin(alpha)) * self.cylinder.length / 2

    def get_surface(self) -> float:
        return 42

    def check_point(self, point) -> bool:
        d = np.dot(point - self.cylinder.center, self.norm)
        # d = np.linalg.norm(np.cross(point - self.cylinder.center, self.cylinder.axis))

        return (d > self.segm_radius) and self.cylinder.check_point(point)

    def generate_point(self) -> np.array:
        inside = False
        while not inside:
            point = self.cylinder.generate_point()
            inside = self.check_point(point)

        return point
