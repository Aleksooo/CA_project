import numpy as np
from tqdm import tqdm
from .pbc import distance_pbc2, delta_pbc

def push_atoms_apart(structure, min_dist2, max_dist2, iteration_lim=10):
    min_dist = np.sqrt(min_dist2)
    max_dist = np.sqrt(max_dist2)

    new_structure = structure.copy()
    ins = 0
    overlap = True
    while overlap and (ins < iteration_lim):
        overlap = False
        overlap_counter = 0
        ins += 1

        print(f"Iteration {ins}")
        for i, atom in enumerate(tqdm(new_structure.atoms[:-1], desc='Atoms')):
            for j, other_atom in enumerate(new_structure.atoms[i+1:]):
                if atom.mol_id != other_atom.mol_id:
                    delta = delta_pbc(atom.xyz, [other_atom.xyz], structure.box)[0]

                    if np.all(abs(delta) < min_dist):
                        # if distance_pbc2(atom.xyz, [other_atom.xyz], structure.box)[0] < min_dist2:
                        overlap = True
                        overlap_counter += 1
                        other_atom.xyz += delta / np.linalg.norm(delta) * \
                            np.random.uniform(min_dist, max_dist)

        print(f"{overlap_counter} overlaps detected")

    return new_structure.apply_pbc()
