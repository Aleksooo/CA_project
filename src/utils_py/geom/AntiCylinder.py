from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .Shape import Shape
from .Box import Box
from .Cylinder import Cylinder

@dataclass
class AntiCylinder(Shape):
    cylinder: Cylinder = field(default_factory=Cylinder)
    box: Box = field(default_factory=Box)

    # radius: float = 0
    # length: float = 0
    # axis: np.array = field(default_factory=np.array)
    # borders_center: np.array = field(default_factory=np.array)
    # borders:  np.array = field(default_factory=np.array)

    def get_volume(self) -> float:
        return self.box.get_volume() - self.cylinder.get_volume()

    def get_surface(self) -> float:
        return self.box.get_surface() + self.cylinder.get_surface()

    def check_point(self, point) -> bool:
        return self.box.check_point(point) and (not self.cylinder.check_point(point))

    def generate_point(self) -> np.array:
        inside_cylinder = True
        while inside_cylinder:
            point = self.box.generate_point()
            inside_cylinder = self.cylinder.check_point(point)
        return point
