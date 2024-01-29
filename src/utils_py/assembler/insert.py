import numpy as np
from scipy.spatial.transform import Rotation
import sys
from .pbc import distance_pbc, distance_pbc2
from .grid import check_grid

def insert_point_into_shape(
    shape, points, point_id, system_size, mol_size, mol_id,
    insertion_limit = int(1e5),
    package = 0.4
):
    if (mol_size < 0.05) :
        mol_size = 0.05

    overlap = True
    insertion_counter = 0

    if not mol_id:
        return shape.generate_point()

    while overlap and (insertion_counter < insertion_limit):
        insertion_counter += 1

        if 5 * insertion_counter % insertion_limit == 0:
            print(f'Please, wait ... mol #{mol_id} is tried to be inserted\n')

        new_point = shape.generate_point()
        overlap = np.sum(
            # Что насчет квадрата тут???
            distance_pbc2(new_point, points[0:point_id, :], system_size) < \
            ((2 - insertion_counter / insertion_limit) * package * mol_size)
        )

    if insertion_counter > insertion_limit:
        sys.exit('Decrease package')

    return new_point

def find_position(
    structure, new_point, atom_id, mol, mol_id,
    # grid, N, dr,
    rotation_limit = 10,
    min_dist2 = 0.08**2
):
    overlap = True
    rotation_counter = 0

    if not mol_id:
        new_mol = mol.copy()
        new_mol = new_mol.set_XYZ(rotate_random(new_mol.atoms_xyz) + new_point)

        return new_mol

    while overlap and (rotation_counter < rotation_limit):
        # overlap = False
        rotation_counter += 1

        new_mol = mol.copy()
        new_mol = new_mol.set_XYZ(rotate_random(new_mol.atoms_xyz) + new_point)

        for atom_xyz in new_mol.atoms_xyz:
            overlap = np.all(distance_pbc2(atom_xyz, structure.atoms_xyz[0:atom_id, :], structure.box) < min_dist2)
            # overlap = check_grid(grid, structure, atom, N, dr, min_dist2)
            if overlap:
                break

    return new_mol

def rotate_random(xyz):
    theta = np.random.random(3)* 2 * np.pi
    rot = Rotation.from_euler('xyz', theta)
    new_xyz = rot.apply(xyz.copy())

    return new_xyz
