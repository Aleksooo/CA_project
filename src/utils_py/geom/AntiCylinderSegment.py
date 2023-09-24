from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .Shape import Shape
from .Box import Box
from .CylinderSegment import CylinderSegment


@dataclass
class AntiCylinderSegment(Shape):
    cylinder_segment: CylinderSegment = field(default_factory=CylinderSegment)
    box: Box = field(default_factory=Box)

    def get_volume(self) -> float:
        return self.box.get_volume() - self.cylinder_segment.get_volume()

    def get_surface(self) -> float:
        return 42

    def check_point(self, point) -> bool:
        return self.box.check_point(point) and (not self.cylinder_segment.check_point(point))

    def generate_point(self) -> np.array:
        inside_cylinder = True
        while inside_cylinder:
            point = self.box.generate_point()
            inside_cylinder = self.cylinder_segment.check_point(point)

        return point
