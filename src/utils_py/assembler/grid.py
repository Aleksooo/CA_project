import numpy as np
from .pbc import apply_pbc_to_points, distance_pbc2

def check_grid(grid, structure, point, N, dr, min_dist2):
    point_pbc = apply_pbc_to_points(point, structure.box)
    id = np.floor(point_pbc / dr).astype(int)
    id -= (id == N)

    for i in range(27):
        id_neigh = id + np.array([i % 3, i // 3, i // 9]) - 1
        id_neigh += N * (id_neigh < 0)
        id_neigh -= N * (id_neigh >= N)
        # id_x = id[0] + (i % 3) - 1
        # id_y = id[1] + (i // 3) - 1
        # id_z = id[2] + (i // 9) - 1
        if grid[id_neigh[0], id_neigh[1], id_neigh[2]] != []:
            atoms_neigh = np.array([structure.atoms[atom_id - 1].xyz for atom_id in grid[id_neigh[0], id_neigh[1], id_neigh[2]]])
            overlap = np.all(distance_pbc2(point_pbc, atoms_neigh, structure.box) < min_dist2)
        else:
            continue

        if overlap:
            return False

    return True

def add_mol_grid(grid, structure, point, atom_id, N, dr):
    point_pbc = apply_pbc_to_points(point, structure.box)
    id = np.floor(point_pbc / dr).astype(int)
    id -= (id == N)

    grid[id[0], id[1], id[2]].append(atom_id)

    return grid
