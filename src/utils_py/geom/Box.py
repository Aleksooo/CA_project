from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from .Shape import Shape

@dataclass
class Box(Shape):
    borders: np.array = field(default_factory=np.array)

    def get_volume(self) -> float:
        return np.prod(self.borders)

    def get_surface(self) -> float:
        return 2 * (self.borders[0] * self.borders[1] + \
                    self.borders[1] * self.borders[2] + \
                    self.borders[0] * self.borders[2])

    def check_point(self, point) -> bool:
        return np.prod(np.abs(point - self.center) < self.borders / 2)

    def generate_point(self) -> np.array:
        return self.borders * np.random.uniform(-0.5, 0.5, 3) + self.center
