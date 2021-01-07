import numpy as np
import pytest

from rotations_and_angular_velocities.rotation_matrix import is_rotation_matrix, rot2, rotation_matrix_inverse, \
    NotARotationMatrix

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
def test_rotation_matrix_inverse(rotation_matrix):
    with pytest.raises(NotARotationMatrix):
        rotation_matrix_inverse(rotation_matrix)
