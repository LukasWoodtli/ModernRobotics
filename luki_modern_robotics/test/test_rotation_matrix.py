import numpy as np
import pytest

from rotations_and_angular_velocities.rotation_matrix import (
    is_rotation_matrix,
    rot2,
    rotation_matrix_inverse,
    NotARotationMatrix, vector_to_so3, NotAVector, so3_to_vector)

TEST_DATA_OK = [
    np.identity(3),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
]

TEST_DATA_NOT_OK = [
    np.array([[0, -1, 0], [1, 1, 0], [1, 0, 1]]),
    np.array([[0, -2, 0], [1, 1, 0], [0, 0, 1]]),
    np.array([[0, -1, 0], [0, 0, -1], [1, 1, 0]]),
    np.array([[0, -1, 0], [0, -1, 0], [1, 0, 0]]),
    np.array([[0, -1], [0, 0], [1, 1]]),
]


def test_identity_matrix():
    np.testing.assert_array_equal(TEST_DATA_OK[0], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


@pytest.mark.parametrize("matrix", TEST_DATA_OK)
def test_is_rotation_matrix(matrix):
    assert is_rotation_matrix(matrix)


@pytest.mark.parametrize("matrix", TEST_DATA_NOT_OK)
def test_is_not_rotation_matrix(matrix):
    assert not is_rotation_matrix(matrix)


def test_rot2():
    np.testing.assert_array_equal([[1, 0], [0, 1]], rot2(0))


@pytest.mark.parametrize("rotation_matrix", TEST_DATA_OK)
def test_rotation_matrix_inverse(rotation_matrix):
    expected_inverse = np.linalg.inv(rotation_matrix)
    expected_transpose = rotation_matrix.T

    inverse = rotation_matrix_inverse(rotation_matrix)

    np.testing.assert_array_equal(expected_inverse, inverse)
    np.testing.assert_array_equal(expected_transpose, inverse)


@pytest.mark.parametrize("rotation_matrix", TEST_DATA_NOT_OK)
def test_rotation_matrix_inverse_raises(rotation_matrix):
    with pytest.raises(NotARotationMatrix):
        rotation_matrix_inverse(rotation_matrix)


def test_vector_to_so3():
    so3 = vector_to_so3(np.array([[1], [2], [3]]))
    expected = [[0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0]]

    np.testing.assert_array_equal(expected, so3)


def test_vector_to_so3_raises():
    with pytest.raises(NotAVector):
        vector_to_so3(np.array([1]))


TEST_DATA_VECTORS =[
    np.array([[1], [2], [3]]),
    np.array([[2], [2], [3]]),
    np.array([[0.2], [1.4], [3.8]])
]


@pytest.mark.parametrize("vector", TEST_DATA_VECTORS)
def test_vector_to_so3_negative_transpose(vector):
    so3 = vector_to_so3(vector)
    expected = np.negative(so3.T)

    np.testing.assert_array_equal(expected, so3)


@pytest.mark.parametrize("vector", TEST_DATA_VECTORS)
@pytest.mark.parametrize("matrix", TEST_DATA_OK)
def test_vector_to_so3_with_rotation_matrix(vector, matrix):
    # R [ω]R^T = [Rω]
    r_omega_r_transpose =\
        np.dot(matrix, np.dot(vector_to_so3(vector), matrix.T))

    r_omega = np.dot(matrix, vector)
    so3_r_omega = vector_to_so3(r_omega)

    np.testing.assert_array_equal(r_omega_r_transpose,
                                  so3_r_omega)


def test_so3_to_vector():
    so3 = np.array([[0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]])

    vector = so3_to_vector(so3)

    expected = [[1], [2], [3]]

    np.testing.assert_array_equal(expected, vector)


@pytest.mark.parametrize("vector", TEST_DATA_VECTORS)
def test_vector_to_so3_and_back(vector):
    so3 = vector_to_so3(vector)
    vector_new = so3_to_vector(so3)

    np.testing.assert_array_equal(vector, vector_new)
