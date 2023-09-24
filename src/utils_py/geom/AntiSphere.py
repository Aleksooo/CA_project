from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from .Shape import Shape
from .Box import Box
from .Sphere import Sphere

@dataclass
class AntiSphere(Shape):
    radius:  float = 1
    borders_center: np.array = field(default_factory=np.array)
    borders:  np.array = field(default_factory=np.array)

    def get_volume(self) -> float:
        return self.get_box().get_volume() - self.get_sphere().get_volume()

    def get_surface(self) -> float:
        return self.get_box().get_surface() + self.get_sphere().get_surface()

    def check_point(self, point) -> bool:
        return self.get_box().check_point(point) and (not self.get_sphere().check_point(point))

    def generate_point(self) -> np.array:
        inside_sphere = True
        while inside_sphere:
            point = self.get_box().generate_point()
            inside_sphere = self.get_sphere().check_point(point)
        return point

    def get_box(self):
        return Box(center=self.borders_center, borders=self.borders)

    def get_sphere(self):
        return Sphere(center=self.center, radius=self.radius)
