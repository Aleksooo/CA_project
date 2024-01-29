from dataclasses import dataclass, field
import numpy as np

@dataclass
class Atom:
    mol_id: int = 1
    mol_name: str = 'MOL'
    name: str = 'ATOM'
    id: int = 1
    xyz: np.array = field(default_factory=np.array)

    def make_gro_line(self) -> str:
        start = f'{self.mol_id:5d}{self.mol_name:<5}{self.name:>5}{self.id%100000:5d}'
        coords = ''.join([f'{x:8.3f}' for x in self.xyz])
        return f'{start}{coords}'

    def copy(self):
        return Atom(
            mol_id = self.mol_id,
            mol_name = self.mol_name,
            name = self.name,
            id = self.id,
            xyz = self.xyz.copy()
        )


@dataclass
class Atom_label:
    mol_id: int = 1
    mol_name: str = 'MOL'
    name: str = 'ATOM'
    id: int = 1

    def copy(self):
        return Atom_label(
            mol_id = self.mol_id,
            mol_name = self.mol_name,
            name = self.name,
            id = self.id
        )

