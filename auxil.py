import numpy as np
from tqdm import tqdm

def y_roll_approx(zeta, l, phi, theta):
    if np.abs(theta - np.pi / 2) < 0.5:
            return (phi * l - (1 - 120 * zeta**2 + 720 * zeta**4) * (theta - np.pi / 2)**3 / 360 - (12 * zeta**2 - 1) * (theta - np.pi / 2) / 6) / 2

    return (phi * l - (2 * theta - np.pi) / 4 / np.cos(theta)**2 + np.sign(2 * theta - np.pi) * np.sqrt(1 / np.cos(theta)**2 - 4 * zeta**2) + np.tan(theta) / 2) / 2

def rho_roll_approx(zeta, l, phi, theta):
    if np.abs(theta - np.pi / 2) < 0.5:
            return phi - (1 - 120 * zeta**2 + 720 * zeta**4) * (theta - np.pi / 2)**3 / 360 / l - (12 * zeta**2 - 1) * (theta - np.pi / 2) / 6 / l

    return (phi * l - (2 * theta - np.pi) / 4 / np.cos(theta)**2 + np.sign(2 * theta - np.pi) * np.sqrt(1 / np.cos(theta)**2 - 4 * zeta**2) + np.tan(theta) / 2) / l

def rho_worm(zeta, l, phi, theta):
    r = np.sqrt(2*phi*l / (2*theta - np.sin(2*theta)))
    zeta_c = 4*r*np.sin(theta)**3 / (3 * (2*theta - np.sin(2*theta)))
    rho = np.nan_to_num(2 * np.sqrt(r**2 - (zeta + zeta_c)**2) / l)
    return rho * np.logical_and(-zeta_c + r * np.cos(theta) < zeta, zeta < r - zeta_c)

def get_center_pbc(positions, box):
    theta = positions / box * 2 * np.pi
    center = np.zeros(3)

    for i in range(3):
        phi = np.cos(theta[:, i])
        psi = np.sin(theta[:, i])

        phi_mean = np.average(phi)
        psi_mean = np.average(psi)

        theta_mean = np.arctan2(-psi_mean, -phi_mean) + np.pi
        center[i] = box[i] * theta_mean / 2 / np.pi

    return center

def apply_pbc(positions, box):
    half_box_size = box / 2

    ids = abs(positions - half_box_size) >= half_box_size
    positions -= np.sign(positions) * box * ids

    return positions

def center_residue(AtomGroup, box):
    positions = AtomGroup.copy().positions
    center = get_center_pbc(positions, box)
    positions -= center
    positions += box / 2

    return apply_pbc(positions, box)

def density_plane(universe, residue_name, M, center=True, begin=0):
    u = universe.copy()
    residue = u.select_atoms(f'resname {residue_name}', updating=True)
    box = u.dimensions[:3] * 0.1 #nm
    traj = u.trajectory[begin:]
    N = len(traj)
    axis_x, dx = np.linspace(0, box[0], M, retstep=True)
    axis_z, dz = np.linspace(0, box[2], M, retstep=True)
    dr = np.array([dx, dz])
    data = np.zeros((N, M, M), dtype=np.float32)

    for ts in tqdm(traj):
        residue.positions *= 0.1
        if center:
            residue.positions = center_residue(residue, box)

        residue_ids = np.floor(residue.positions[:, ::2] / dr).astype(int)
        residue_ids -= (residue_ids == M)

        for id in residue_ids:
            data[ts.frame-begin, id[0], id[1]] += 1

    data = np.mean(data, axis=0)

    return axis_x, axis_z, data / (dx * dz * box[0]), dx, dz

def get_border(grid, min=3, max=7):
    border = np.zeros_like(grid)
    shape = grid.shape
    # for (i, j) in np.ndindex((shape[0]-2, shape[1]-2)):
    #     # neigh = (grid[i-1, j+1] > 0) + (grid[i, j+1] > 0) + (grid[i+1, j+1] > 0) + \
    #     #         (grid[i-1, j] > 0)   +                    + (grid[i+1, j] > 0) + \
    #     #         (grid[i-1, j-1] > 0) + (grid[i, j-1] > 0) + (grid[i+1, j-1] > 0)

    #     # neigh_list = np.array([grid[i-1, j+1], grid[i, j+1], grid[i+1, j+1], grid[i-1, j], grid[i+1, j], grid[i-1, j-1], grid[i, j-1], grid[i+1, j-1]])
    #     neigh_list = np.array([grid[i, j], grid[i+1, j], grid[i+2, j], grid[i, j+1], grid[i+2, j+1], grid[i, j+1], grid[i+1, j+1], grid[i+2, j+1]])
    #     neigh = np.sum(neigh_list > 0)

    #     if min <= neigh <= max:
    #         border[i, j] = 1

    N = shape[0]
    M = shape[1]

    # Определение правой границы
    right_border = np.zeros(N, dtype=int) * M//2
    for j in range(M//2+1, M):
        if not np.any(grid[j, :] > 0): break # Прерывание, если вышли за пределы капли
        right_border[grid[j, :] > 0] = j

    # Определение левой границы
    left_border = np.zeros(N, dtype=int) * M//2
    for j in range(M//2-1, -1, -1):
        if not np.any(grid[j, :] > 0): break # Прерывание, если вышли за пределы капли
        left_border[grid[j, :] > 0] = j

    border[right_border, np.arange(N, dtype=int)] = 1
    border[left_border, np.arange(N, dtype=int)] = 1

    return border

def get_points_from_border(border):
    points = list()

    for (i, j) in np.ndindex(border.shape):
        if border[i, j]:
            points.append(np.array([i, j]))

    return np.array(points)

def points2xyz(points_grid, dr, box):
    points = np.zeros_like(points_grid, dtype=np.float64)
    for i in range(points_grid.shape[0]):
        points[i, :] = (points_grid[i, :] + 0.5) * dr

    return points - box / 2
