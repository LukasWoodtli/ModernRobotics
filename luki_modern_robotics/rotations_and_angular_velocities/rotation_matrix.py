"""Code for rotation matrices and angular velocities."""
from math import cos, sin
import numpy as np

EXPECTED_VECTOR_DIMENSION = 3


class NotARotationMatrix(Exception):
    pass


class NotAVector(Exception):
    pass


class NotAUnitVector(Exception):
    pass


class NotAso3Matrix(Exception):
    pass


def rot2(phi):
    """
    Create a 2D rotation matrix with a given angle.
    :param phi: Angle in radians.
    :return: The 2D rotation matrix (SO(2))
    """
    return np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])


def is_rotation_matrix(mat: np.array):
    """
    Check if the given array is an rotation matrix.
    :param mat: The matrix to check.
    :return: True if the matrix is a rotation matrix, False otherwise.
    """

    is_square = _is_square(mat)
    if not is_square:
        return False

    orthogonal_unit_vecs = _orthogonal_column_vectors(mat)
    right_handed = _is_right_handed_frame(mat)

    is_rot = orthogonal_unit_vecs and right_handed

    if is_rot:
        # determinant of a rotation matrix is 1
        assert np.linalg.det(mat)

    return is_rot


def RotInv(rotation_matrix):
    """Returns the inverse of an rotation matrix."""
    if not is_rotation_matrix(rotation_matrix):
        raise NotARotationMatrix

    return rotation_matrix.T


def VecToso3(vec):
    shape = vec.shape[0]
    if shape is not EXPECTED_VECTOR_DIMENSION:
        raise NotAVector

    x1, x2, x3 = vec
    # x1 = x1[0]
    # x2 = x2[0]
    # x3 = x3[0]

    skew_symmetric_matrix = np.array([
        [0, -x3, x2],
        [x3, 0, -x1],
        [-x2, x1, 0]
    ], dtype=object)

    assert _is_skew_symmetric(skew_symmetric_matrix)
    return skew_symmetric_matrix


def so3ToVec(so3):
    if not _is_skew_symmetric(so3):
        raise NotAso3Matrix

    vector = np.array([so3[2][1],
                       so3[0][2],
                       so3[1][0]])

    return vector


def AxisAng3(expc3):
    """
    Extracts the rotation axis omega_hat and the rotation amount theta
    from the 3-vector omega_hat theta of exponential coordinates for rotation, `expc3`
    """
    shape = expc3.shape[0]
    if shape is not EXPECTED_VECTOR_DIMENSION:
        raise NotAVector

    theta = np.linalg.norm(expc3)
    omega_hat = expc3 / theta
    return omega_hat, theta


def MatrixExp3(so3mat):
    """
    Calculate matrix exponential for rotation matrix
    using Rodrigues’ formula.
    """
    if not _is_skew_symmetric(so3mat):
        raise NotAso3Matrix

    # could check for near to zero here

    omega_theta = so3ToVec(so3mat)
    omega_hat, theta = AxisAng3(omega_theta)

    return rodrigues_formula(omega_hat, theta)


def rodrigues_formula(omega_hat, theta):
    omega_hat_so3 = VecToso3(omega_hat)
    R = np.eye(3) + np.sin(theta) * omega_hat_so3 + \
        (1 - np.cos(theta)) * np.dot(omega_hat_so3, omega_hat_so3)
    return R


def MatrixLog3(R):
    if not is_rotation_matrix(R):
        raise NotARotationMatrix

    if np.isclose(R,
                  np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])).all():
        so3mat = np.zeros([3, 3])
    elif np.isclose(np.trace(R), -1.0):
        theta = np.pi

        # Other cases should be handled here (see original MR library)
        omega_hat = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
            * np.array([R[0][2], R[1][2], 1 + R[2][2]])

        so3mat = VecToso3(theta * omega_hat)
    else:
        theta = np.arccos((np.trace(R) -1) / 2.0)
        omega_hat_so3 = 1.0 / (2 * np.sin(theta)) * (R - R.T)
        so3mat = theta * omega_hat_so3
        assert _is_skew_symmetric(so3mat)

    return so3mat


def RpToTrans(R, p):
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]


def TransToRp(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p


def TransInv(T):
    R, p = TransToRp(T)
    R_T = RotInv(R)
    first_row = np.c_[R_T, -np.dot(R_T, p)]
    return np.r_[first_row, [[0, 0, 0, 1]]]


def VecTose3(V):

    return np.r_[
        np.c_[VecToso3(np.array([[V[0]], [V[1]], [V[2]]])),
              [V[3], V[4], V[5]]],
        np.zeros((1, 4))]


def se3ToVec(se3):
    so3 = np.array([i[:3] for i in se3[:3]])
    omega_vec = so3ToVec(so3)
    v_vec = [i[3] for i in se3[:3]]
    return [*omega_vec, *v_vec]


def Adjoint(T):
    R, p = TransToRp(T)
    p_so3 = VecToso3(p)
    p_so3_R = np.dot(p_so3, R)

    adj = np.r_[
        np.c_[R, np.zeros((3, 3))],
        np.c_[p_so3_R, R]]

    return adj


def ScrewToAxis(q, s, h):
    """
    Returns normalized screw axis S
    :param q: location of direction vector s
    :param s: unit vector in direction of screw S
    :param h: pitch of the screw
    :return: normalized screw axis
    """
    if not _is_unit_vector(s):
        raise NotAUnitVector()

    return np.r_[s,
                 np.cross(-s, q) + np.dot(h, s)]

def AxisAng6(expc6):
    """
    Extracts normalized screw axis S and distance traveled along the screw (theta)
    :param expc6: 6-vector of exponential coordinates for rigid-body motion S*tetha
    :return: S and theta
    """
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if np.isclose(theta, 0):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return np.array(expc6 / theta), theta


def MatrixExp6(se3mat):
    """
    Computes homogeneous transformation matrix T corresponding to matrix exponential of se3mat
    :param se3mat: in se(3)
    :return: T in SE(3)
    """
    se3mat = np.array(se3mat)
    R = se3mat[0:3, 0:3]
    omegatheta = so3ToVec(R)
    vtheta = se3mat[0: 3, 3]
    if np.isclose(np.linalg.norm(omegatheta), 0):
        return np.r_[np.c_[np.eye(3), vtheta], [[0, 0, 0, 1]]]

    _omega_hat, theta = AxisAng3(omegatheta)
    omgmat = R / theta
    assert _is_skew_symmetric(omgmat)
    e_omg_theta = MatrixExp3(R)
    return np.r_[np.c_[e_omg_theta,
                        np.dot(
                            np.eye(3) * theta + \
                            (1 - np.cos(theta)) * omgmat + \
                            (theta - np.sin(theta)) * \
                            np.dot(omgmat, omgmat),
                            vtheta) / theta],
                        [[0, 0, 0, 1]]]



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


def _is_skew_symmetric(skew_symmetric_matrix):
    negative_transpose = np.negative(skew_symmetric_matrix.T)
    return (skew_symmetric_matrix == negative_transpose).all()


def _is_unit_vector(v):
    return np.isclose(1, np.linalg.norm(v))
