from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Shape(ABC):
    center: np.array = field(default_factory=np.array)
    allowed_mol_names: str = ''
    
    @abstractmethod
    def get_volume(self) -> float:
        pass
    
    @abstractmethod
    def get_surface(self) -> float:
        pass
    
    @abstractmethod
    def check_point(self) -> bool:
        pass
    
    @abstractmethod
    def generate_point(self) -> np.array:
        pass
        