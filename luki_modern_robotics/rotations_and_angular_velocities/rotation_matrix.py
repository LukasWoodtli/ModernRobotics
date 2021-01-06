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


def is_rotation_matrix(mat: np.array):
    """
    Check if the given array is an rotation matrix.
    :param mat: The matrix to check.
    :return: True if the matrix is a rotation matrix, False otherwise.
    """

    is_square = _is_square(mat)
    if not is_square:
        return

    orthogonal_unit_vecs = _orthogonal_column_vectors(mat)
    right_handed = _is_right_handed_frame(mat)

    return orthogonal_unit_vecs and right_handed


def _is_square(m):
    """
    Check if a matrix is square (N x N).
    :param m: Matrix to check.
    :return: True if matrix is square, False otherwise.
    """
    return all(len(row) == len(m) for row in m)


def _is_right_handed_frame(mat):
    """
    Check if right handed frame
    :param mat: Matrix representing a frame
    :return: True if matrix represents a right handed frame
    """
    right_handed = np.isclose(np.linalg.det(mat), 1)
    return right_handed


def _orthogonal_column_vectors(mat):
    """
    check if columns are orthogonal unit vectors
    R^TR = 1
    :param mat: matrix to test
    :return: True if column vector are orthogonal and have unit length
    """
    return np.isclose(mat.T @ mat, np.identity(mat.shape[0])).all()
