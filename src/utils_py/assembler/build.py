import numpy as np
from tqdm import tqdm
import sys
from src.utils_py.io.gro import read_gro
from src.utils_py.assembler.insert import insert_point_into_shape, find_position
from .grid import add_mol_grid

def build_system(
    dir, structure, names=None, numbers=None, shapes=None, substr_points=None,
    insertion_limit = int(1e5),
    rotation_limit = 10,
    package = 0.4,
    min_dist2 = 0.08**2
):
    if names == None or numbers == None or shapes == None:
        sys.exit('Please, give all the parametrs')

    if not(len(names) == len(numbers) == len(shapes)):
        sys.exit('Parameters have different lengths')

    system_size = structure.box
    print('Number of molecules:')
    for i, name in enumerate(names):
        print(name, '\t', numbers[i])

    atoms_number = []
    for i, name in enumerate(names):
        atoms_number.append(len(read_gro(f'{dir}/gro/{name}.gro').atoms))
    atoms_number = np.array(atoms_number)
    structure.atoms = np.hstack((structure.atoms, np.empty(np.sum(atoms_number * numbers))))
    structure.atoms_xyz = np.vstack((structure.atoms_xyz, np.zeros((np.sum(atoms_number * numbers), 3))))

    points = np.vstack((substr_points, np.zeros((sum(numbers), 3))))
    point_id = substr_points.shape[0]
    atom_id = substr_points.shape[0]

    print('\nFilling system:')
    for i, name in enumerate(names):
        mol = read_gro(f'{dir}/gro/{name}.gro').center_atoms_to_center().center_atoms_to_zero()
        mol_size = np.max(np.linalg.norm(mol.atoms_xyz, axis=1))

        for mol_id in tqdm(range(numbers[i])):
            new_point = insert_point_into_shape(shapes[i], points, point_id, system_size, mol_size, mol_id, insertion_limit, package)
            points[point_id, :] = new_point
            point_id += 1

            new_mol = find_position(structure, new_point, atom_id, mol, mol_id, rotation_limit, min_dist2)
            for j, atom_label in np.ndenumerate(new_mol.atoms):
                new_atom_label = atom_label.copy()
                new_atom_label.id = atom_id + 1
                new_atom_label.mol_id = mol_id + 1

                structure.add_atom(new_atom_label, new_mol.atoms_xyz[j, :],  atom_id)
                atom_id += 1

    return structure
