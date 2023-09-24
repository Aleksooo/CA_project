from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from .Shape import Shape

@dataclass
class Roll(Shape):
    phi: float = 0
    theta: float = 0
    borders: np.array = field(default_factory=np.array)

    def get_volume(self) -> float:
        return self.phi * np.prod(self.borders)

    def get_surface(self) -> float:
        return 42

    def y(self, zeta):
        l = (self.borders[0] + self.borders[1]) / self.borders[2] / 2
        if np.abs(self.theta - np.pi / 2) < 0.5:
                return (l * self.phi - (1 - 120 * zeta**2 + 720 * zeta**4) * (self.theta - np.pi / 2)**3 / 360 - (12 * zeta**2 - 1) * (self.theta - np.pi / 2) / 6) / 2

        return (self.phi * l - (2 * self.theta - np.pi) / 4 / np.cos(self.theta)**2 + np.sign(2 * self.theta - np.pi) * np.sqrt(1 / np.cos(self.theta)**2 - 4 * zeta**2) + np.tan(self.theta) / 2) / 2

    def check_point(self, point) -> bool:
        return np.abs(point[1]/self.borders[2]) <= self.y(point[2]/self.borders[2])

    def generate_point(self) -> np.array:
        Y_max = (self.y(0) if (self.theta > np.pi/2) else self.y(0.5)) * self.borders[2]
        point = np.array([np.random.uniform(-self.borders[0]/2, self.borders[0]/2),
                          np.random.uniform(-Y_max, Y_max),
                          np.random.uniform(-self.borders[2]/2, self.borders[2]/2)])
        while not self.check_point(point):
            point = np.array([np.random.uniform(-self.borders[0]/2, self.borders[0]/2),
                              np.random.uniform(-Y_max, Y_max),
                              np.random.uniform(-self.borders[2]/2, self.borders[2]/2)])

        return point + self.center
