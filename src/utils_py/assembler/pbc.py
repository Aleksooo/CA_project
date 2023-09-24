import numpy as np


def delta_pbc(point1: np.array, points: np.array, box_size: np.array) -> np.array:
    delta = points - point1
    idx_pbc = np.abs(delta) * 2 >= box_size
    delta -= np.sign(delta) * box_size * idx_pbc

    return delta

def distance_pbc2(point1: np.array, points: np.array, box_size: np.array) -> np.array:
    return np.sum(delta_pbc(point1, points, box_size)**2, axis=1)

def distance_pbc(point1: np.array, points: np.array, box_size: np.array) -> np.array:
    return np.sqrt(distance_pbc2(point1, points, box_size))

def apply_pbc_to_points(points: np.array, box_size: np.array):
    points_pbc = points.copy()
    half_box_size = box_size / 2

    idx_pbc = abs(points_pbc - half_box_size) >= half_box_size
    points_pbc -= np.sign(points_pbc) * box_size * idx_pbc

    return points_pbc

