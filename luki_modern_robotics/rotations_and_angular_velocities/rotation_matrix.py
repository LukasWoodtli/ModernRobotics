"""Code for rotation matrices and angular velocities."""
from math import cos, sin
import numpy as np


def rot2(phi):
    """
    Create a 2D rotation matrix with a given angle.
    :param phi: Angle in radians.
    :return: The 2D rotation matrix (SO(2))
    """
    return np.array([cos(phi), -sin(phi)], [sin(phi), cos(phi)])


def is_square(m):
    """
    Check if a matrix is square (N x N).
    :param m: Matrix to check.
    :return: True if matrix is square, False otherwise.
    """
    return all(len(row) == len(m) for row in m)


def is_rotation_matrix(mat: np.array):
    """
    Check if the given array is an rotation matrix.
    :param mat: The matrix to check.
    :return: True if the matrix is a rotation matrix, False otherwise.
    """

    if not is_square(mat):
        return False

    # check if columns are orthogonal
    # unit vectors
    # R^TR = 1
    orthogonal_unit_vecs = np.isclose(mat.T@mat, np.identity(mat.shape[0]))

    # check if right handed frame
    right_handed = np.isclose(np.linalg.det(mat), 1)

    return orthogonal_unit_vecs and right_handed
