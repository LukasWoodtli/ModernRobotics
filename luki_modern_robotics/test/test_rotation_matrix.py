import numpy as np
import pytest

from rotations_and_angular_velocities.rotation_matrix import (
    is_rotation_matrix, rot2, RotInv,
    NotARotationMatrix, VecToso3,
    NotAVector, so3ToVec, AxisAng3,
    MatrixExp3, MatrixLog3, RpToTrans,
    TransToRp, TransInv, VecTose3, se3ToVec,
    Adjoint, ScrewToAxis, AxisAng6,
    MatrixExp6, _is_skew_symmetric)

from rotations_and_angular_velocities.rotation_matrix import rodrigues_formula


TEST_DATA_OK = [
    np.identity(3),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
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
    np.array([1, 2, 3]),
    np.array([2, 2, 3]),
    np.array([0.2, 1.4, 3.8])
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

    expected = [1, 2, 3]

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


def test_rodrigues_formula():
    # example 3.12 from book
    omega_hat = np.array([ 0, 0.866,  0.5])
    theta = 0.524

    expected = np.array([[0.865830625594382, -0.25017371513495706,  0.4333008746137456],
                  [0.25017371513495706, 0.9664561804705362,  0.05809789542503132],
                  [-0.4333008746137456,  0.05809789542503132,  0.8993744451238458]])

    matrix_exp = rodrigues_formula(omega_hat, theta)

    np.testing.assert_array_almost_equal(expected, matrix_exp)


def test_MatrixLog3():
    R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    expected = np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])

    so3mat = MatrixLog3(R)

    np.testing.assert_array_almost_equal(expected, so3mat)


def test_MatrixLog3_identity():
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    np.testing.assert_array_equal(R, np.diag(np.full(3, 1)))

    expected = np.zeros([3, 3])

    so3mat = MatrixLog3(R)

    np.testing.assert_array_almost_equal(expected, so3mat)


def test_RpToTrans():
    R = np.array([[1, 0,  0],
                  [0, 0, -1],
                  [0, 1,  0]])
    p = np.array([1, 2, 5])

    expected = np.array([[1, 0,  0, 1],
              [0, 0, -1, 2],
              [0, 1,  0, 5],
              [0, 0,  0, 1]])

    trans = RpToTrans(R, p)

    np.testing.assert_array_equal(expected, trans)


def test_TransToRp():
    T = np.array([[1, 0,  0, 1],
              [0, 0, -1, 2],
              [0, 1,  0, 5],
              [0, 0,  0, 1]])

    expected_R = np.array([[1, 0,  0],
                  [0, 0, -1],
                  [0, 1,  0]])
    expected_p = np.array([1, 2, 5])

    R, p = TransToRp(T)

    np.testing.assert_array_equal(expected_R, R)
    np.testing.assert_array_equal(expected_p, p)


def test_TransInv():
    T = np.array([[1, 0,  0, 0],
                  [0, 0, -1, 0],
                  [0, 1,  0, 3],
                  [0, 0,  0, 1]])

    expected = np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])

    T_inv = TransInv(T)

    np.testing.assert_array_equal(expected, T_inv)


def test_VecTose3():
    V = np.array([1, 2, 3, 4, 5, 6])
    expected = np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])

    se3 = VecTose3(V)

    np.testing.assert_array_equal(expected, se3)


def test_se3ToVec():
    V = np.array([[0, -3, 2, 4],
                  [3, 0, -1, 5],
                  [-2, 1, 0, 6],
                  [0, 0, 0, 0]])
    expected = np.array([1, 2, 3, 4, 5, 6])

    vec = se3ToVec(V)

    np.testing.assert_array_equal(expected, vec)


def test_Adjoint():
    T = np.array([[1, 0,  0, 0],
                  [0, 0, -1, 0],
                  [0, 1,  0, 3],
                  [0, 0,  0, 1]])

    expected = np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])

    adj = Adjoint(T)

    np.testing.assert_array_equal(expected, adj)


def test_ScrewToAxis():
    q = np.array([3, 0, 0])
    s = np.array([0, 0, 1])
    h = 2
    S = ScrewToAxis(q, s, h)

    expected = np.array([0, 0, 1, 0, -3, 2])
    np.testing.assert_array_equal(expected, S)

def test_AxisAng6():
    expc6 = np.array([1, 0, 0, 1, 2, 3])

    S, theta = AxisAng6(expc6)

    expected_S = np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    expected_theta = 1.0

    np.testing.assert_array_equal(expected_S, S)
    assert expected_theta == theta


def test_MatrixExp6():
    se3mat = np.array([[0,          0,           0,          0],
                       [0,          0, -1.57079632, 2.35619449],
                       [0, 1.57079632,           0, 2.35619449],
                       [0,          0,           0,          0]])

    expected_T = np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])

    T = MatrixExp6(se3mat)

    np.testing.assert_array_almost_equal(expected_T, T)


def test_various_2():
    R_sb = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, -1, 0]])
    assert is_rotation_matrix(R_sb)
    R_sb_inv = np.linalg.inv(R_sb)
    assert is_rotation_matrix(R_sb_inv)
    R_sb_inv_expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_array_equal(R_sb_inv_expected, R_sb_inv)


def test_various_3():
    R_as = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    R_sb = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, -1, 0]])
    assert is_rotation_matrix(R_as)
    R_ab = R_as @ R_sb
    assert is_rotation_matrix(R_ab)
    np.testing.assert_array_equal(R_ab, np.array([[0, -1, 0],
                                         [1, 0, 0],
                                         [0, 0, 1]]))

def test_various_5():
    R_sb = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, -1, 0]])
    p_b = np.array([1, 2, 3]).T
    p_s = R_sb @ p_b
    np.testing.assert_array_equal(p_s, np.array([1, 3, -2]))


def test_various_7():
    R_as = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    omega_s = np.array([3, 2, 1]).T
    omega_a = R_as @ omega_s
    np.testing.assert_array_equal(omega_a, np.array([1, 3, 2]))


def test_various_8():
    R_sa = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    res = MatrixLog3(R_sa)
    np.testing.assert_array_almost_equal(res, np.array(
         [[0., 1.2092, -1.2092],
          [-1.2092,  0.,  1.2092],
          [1.2092, -1.2092,  0.]]))
    _omghat, theta = AxisAng3(res)
    assert theta == 2.961921958772245


def test_various_9():
    omega_hat_theta = np.array([1, 2, 0]).T
    omega_hat_theta = VecToso3(omega_hat_theta)
    res = MatrixExp3(omega_hat_theta)
    np.testing.assert_array_almost_equal(res, np.array(
        [[-0.29381830116573315, 0.6469091505828666, 0.7036898157513979],
         [0.6469091505828666, 0.6765454247085667, -0.35184490787569894],
         [-0.7036898157513979, 0.35184490787569894, -0.6172728764571664]]))


def test_various_10():
    omega = np.array([1, 2, 0.5]).T
    res = VecToso3(omega)
    np.testing.assert_array_equal(res, np.array(
        [[0., -0.5, 2.],
         [0.5, 0., -1.],
         [-2., 1., 0.]]))


def test_various_11():
    omega_hat_theta = np.array([[0, 0.5, -1], [-0.5, 0, 2], [1, -2, 0]])
    assert _is_skew_symmetric(omega_hat_theta)
    res = MatrixExp3(omega_hat_theta)
    np.testing.assert_array_almost_equal(res, np.array(
        [[0.6048204475307473, 0.7962739995355433, -0.011829789194075735],
         [0.4683005683660654, -0.3436104783954592, 0.8140186833266569],
         [0.6441170731448801, -0.4978750413512547, -0.5807182098770107]]))


def test_various_12():
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    omega_hat_theta = MatrixLog3(R)
    np.testing.assert_array_almost_equal(omega_hat_theta, np.array(
        [[0., 1.2092, 1.2092],
         [-1.2092, 0., 1.2092],
         [-1.2092, -1.2092, 0.]]))
