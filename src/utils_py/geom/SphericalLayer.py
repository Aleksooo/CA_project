from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import distance
from .Shape import Shape

@dataclass
class SphericalLayer(Shape):
    outer_radius:  float = 0
    inner_radius: float = 1
    
    def get_volume(self) -> float:
        return 4/3*np.pi*(self.outer_radius**3 - self.inner_radius**3)
    
    def get_surface(self) -> float:
        return 4*np.pi*(self.outer_radius**2 + self.inner_radius**2)
    
    def check_point(self, point) -> bool:
        return self.inner_radius < distance.euclidean(point, self.center) < self.outer_radius
    
    def generate_point(self) -> np.array:
        r = np.random.uniform(self.inner_radius, self.outer_radius)
        phi = np.random.uniform(0, 2*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        return r * np.array([
            sin_theta * np.cos(phi), 
            sin_theta * np.sin(phi),
            cos_theta
            ]) + self.center
        