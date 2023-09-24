from dataclasses import dataclass, field
import numpy as np

@dataclass
class Molecule:
    id: int = 1
    name: str = 'MOL'
    center: np.array = field(default_factory=list)
    atoms: list = field(default_factory=list)

    def get_xyz(self) -> np.array:
        return np.array([atom.xyz for atom in self.atoms])

    def get_center(self) -> np.array:
        return np.average(self.get_XYZ(), axis=0)

