import numpy as np
import pytest

from rotations_and_angular_velocities.rotation_matrix import (
    is_rotation_matrix,
    rot2,
    RotInv,
    NotARotationMatrix, VecToso3, NotAVector, so3ToVec, AxisAng3, MatrixExp3)

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

    inverse = RotInv(rotation_matrix)

    np.testing.assert_array_equal(expected_inverse, inverse)
    np.testing.assert_array_equal(expected_transpose, inverse)


@pytest.mark.parametrize("rotation_matrix", TEST_DATA_NOT_OK)
def test_rotation_matrix_inverse_raises(rotation_matrix):
    with pytest.raises(NotARotationMatrix):
        RotInv(rotation_matrix)


def test_vector_to_so3():
    so3 = VecToso3(np.array([[1], [2], [3]]))
    expected = [[0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0]]

    np.testing.assert_array_equal(expected, so3)


def test_vector_to_so3_raises():
    with pytest.raises(NotAVector):
        VecToso3(np.array([1]))


TEST_DATA_VECTORS = [
    np.array([[1], [2], [3]]),
    np.array([[2], [2], [3]]),
    np.array([[0.2], [1.4], [3.8]])
]


@pytest.mark.parametrize("vector", TEST_DATA_VECTORS)
def test_vector_to_so3_negative_transpose(vector):
    so3 = VecToso3(vector)
    expected = np.negative(so3.T)

    np.testing.assert_array_equal(expected, so3)


@pytest.mark.parametrize("vector", TEST_DATA_VECTORS)
@pytest.mark.parametrize("matrix", TEST_DATA_OK)
def test_vector_to_so3_with_rotation_matrix(vector, matrix):
    # R [ω]R^T = [Rω]
    r_omega_r_transpose = \
        np.dot(matrix, np.dot(VecToso3(vector), matrix.T))

    r_omega = np.dot(matrix, vector)
    so3_r_omega = VecToso3(r_omega)

    np.testing.assert_array_equal(r_omega_r_transpose,
                                  so3_r_omega)


def test_so3_to_vector():
    so3 = np.array([[0, -3, 2],
                    [3, 0, -1],
                    [-2, 1, 0]])

    vector = so3ToVec(so3)

    expected = [[1], [2], [3]]

    np.testing.assert_array_equal(expected, vector)


@pytest.mark.parametrize("vector", TEST_DATA_VECTORS)
def test_vector_to_so3_and_back(vector):
    so3 = VecToso3(vector)
    vector_new = so3ToVec(so3)

    np.testing.assert_array_equal(vector, vector_new)


def test_axis_ang_3():
    omega_hat_theta = np.array([[1], [1], [1]])
    omega_hat, theta = AxisAng3(omega_hat_theta)

    assert np.math.sqrt(3) == theta
    entry = np.math.sqrt(1 / 3)
    np.testing.assert_array_almost_equal([[entry], [entry], [entry]], omega_hat)


def test_axis_ang_3_from_mr_code():
    expc3 = np.array([1, 2, 3])
    omega_hat, theta = AxisAng3(expc3)

    expected_omega_hat = np.array([0.26726124, 0.53452248, 0.80178373])
    expected_theta = 3.7416573867739413

    np.testing.assert_array_almost_equal(expected_omega_hat, omega_hat)
    np.testing.assert_array_almost_equal(expected_theta, theta)


def test_MatrixExp3():
    so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])

    expected = np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [0.69297817,  0.6313497,  0.34810748]])

    matrix_exp = MatrixExp3(so3mat)

    np.testing.assert_array_almost_equal(expected, matrix_exp)
