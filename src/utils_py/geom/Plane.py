from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import sys
from .Shape import Shape

@dataclass
class Plane(Shape):
    norm: np.array = field(default_factory=np.array)

    def get_volume(self) -> float:
        return 0

    def get_surface(self) -> float:
        return 0

    def check_point(self, point) -> bool:
        return np.dot(point - self.center, self.norm) > 0

    def generate_point(self) -> np.array:
        sys.exit('Can`t generate point. Use Box shape instead')
