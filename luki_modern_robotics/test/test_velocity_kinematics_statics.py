import numpy as np
import pytest as pytest
from modern_robotics import JacobianSpace, JacobianBody, IKinSpace, se3ToVec, MatrixLog6, TransInv, FKinBody

from sympy import Matrix
from sympy.abc import x, y

def test_wrench():
    #tau = J^T(phi)*f_{tip}
    f_tip = np.array([0, 0, 0, 2, 0, 0])

    omg_1 = np.array([0, 0, 1])
    q_1 = np.array([0, 0, 0])
    v_1 = np.cross(-omg_1, q_1)
    S_1 = np.r_[omg_1, v_1].T

    omg_2 = np.array([0, 0, 1])
    q_2 = np.array([1, 0, 0])
    v_2 = np.cross(-omg_2, q_2)
    S_2 = np.r_[omg_2, v_2].T

    omg_3 = np.array([0, 0, 1])
    q_3 = np.array([2, 0, 0])
    v_3 = np.cross(-omg_3, q_3)
    S_3 = np.r_[omg_3, v_3].T

    slist = np.c_[S_1, S_2, S_3]
    theta = np.array([0, np.pi/4, -np.pi/4])
    # use library function
    J = JacobianSpace(slist, theta)

    tau = np.dot(J.T, f_tip)
    np.testing.assert_array_almost_equal(tau, [0.      , 0.      , 1.414214])

def test_wrench2():
    # 2
    L1=L2=L3=L4=1
    Theta1 = Theta2 = 0
    Theta3 = np.pi / 2.
    Theta4 = -np.pi / 2.
    F_b = np.array(
        [10, 10, 10]).T

    s4 = np.sin(Theta4)
    s34 = np.sin(Theta3 + Theta4)
    s234 = np.sin(Theta2 + Theta3 + Theta4)
    c4 = np.cos(Theta4)
    c34 = np.cos(Theta3 + Theta4)
    c234 = np.cos(Theta2 + Theta3 + Theta4)

    J_b = np.array([
        [1, 1, 1, 1],
        [L3*s4 + L2*s34 + L1*s234, L3*s4+L2*s34, L3*s4, 0],
        [L4 + L3*c4 + L2*c34 + L1*c234, L4 + L3*c4+L2*c34, L4 + L3*c4, L4]
    ])

    tau = np.dot(J_b.T, F_b)
    np.testing.assert_array_almost_equal(tau, [30., 20., 10., 20.])

def test_jacobian_space():
    S_1 = np.array([0, 0, 1, 0, 0, 0]).T
    S_2 = np.array([1, 0, 0, 0, 2, 0]).T
    S_3 = np.array([0, 0, 0, 0, 1, 0]).T

    slist = np.c_[S_1, S_2, S_3]
    theta = np.array([np.pi/2., np.pi/2., 1])

    J = JacobianSpace(slist, theta)

    print(J)
    np.testing.assert_array_almost_equal(J,
        [[0., 0.,  0.],
         [ 0.,  1.,  0.],
         [ 1.,  0.,  0.],
         [ 0., -2., 0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  1.]])

def test_jacobian_body():
    B_1 = np.array([0, 1, 0, 3, 0, 0]).T
    B_2 = np.array([-1, 0, 0, 0, 3, 0]).T
    B_3 = np.array([0, 0, 0, 0, 0, 1]).T

    blist = np.c_[B_1, B_2, B_3]
    theta = np.array([np.pi/2., np.pi/2., 1])

    J = JacobianBody(blist, theta)

    print(J)
    np.testing.assert_array_almost_equal(J,
        [[0., -1.,  0.],
         [0.,  0.,  0.],
         [1., 0., 0.],
         [0.,  0.,  0.],
         [0., 4., 0.],
         [0.,  0.,  1.]])

def test_manipulability():
    j_b = np.array(
    [[0., -1., 0., 0., -1., 0., 0.],
     [0., 0., 1., 0., 0., 1., 0.],
     [1., 0., 0., 1., 0., 0., 1.],
     [-0.105, 0., 0.006, -0.045, 0., 0.006, 0.],
     [-0.889, 0.006, 0., -0.844, 0.006, 0., 0.],
     [0., -0.105, 0.889, 0., 0., 0., 0.]])

    j_v = j_b[3:]
    a = np.dot(j_v,  j_v.T)
    eigen_values, eigen_vectors = np.linalg.eig(a)

    # get longest axis:
    # https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt/8093043#8093043
    eigen_values = np.sqrt(eigen_values)
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    longest_axis_len = eigen_values[0]
    eigen_vectors = eigen_vectors[:, idx]
    longest_axis_vec = eigen_vectors[:, 0]
    assert longest_axis_len == pytest.approx(1.230535714784298)
    np.testing.assert_array_almost_equal(longest_axis_vec, [ 0.08715237,  0.9961950, -0.0002282767])


def test_newton_raphson_2_iterations():
    # Jacobian of the given matrix
    # Maxima syntax: `jacobian([[x^2-9],[y^2-4]], [x,y]);`
    f = Matrix([x**2 - 9, y**2 -4])
    xy = Matrix([x, y])
    J = f.jacobian(xy)
    assert J == Matrix([[2*x, 0], [0, 2*y]])

    def f(m):
        x = m[0]
        y = m[1]
        return np.array([x**2 - 9, y**2 -4])
    def J_penrose_moore_inverse(m):
        x, y = m[0], m[1]
        res = np.array([[2 * x, 0],
                        [0, 2 * y]])
        res = np.linalg.pinv(res)
        return res


    # initial guess (iteration 0)
    m = (x0, y0) = (1, 1)

    for i in range(2):
        e = 0 - f(m)
        m = m + np.dot(J_penrose_moore_inverse(m), e)
    np.testing.assert_array_equal([3.4,  2.05], m)


def test_inverse_kinematic():
    # joint screws
    omg_0 = np.array([0, 0, 1])
    q_0 = np.array([0, 0, 0])
    v_0 = -np.cross(omg_0, q_0)

    omg_1 = np.array([0, 0, 1])
    q_1 = np.array([1, 0, 0])
    v_1 = -np.cross(omg_1, q_1)

    omg_2 = np.array([0, 0, 1])
    q_2 = np.array([2, 0, 0])
    v_2 = -np.cross(omg_2, q_2)

    s0 = np.r_[omg_0, v_0]
    s1 = np.r_[omg_1, v_1]
    s2 = np.r_[omg_2, v_2]
    Slist = np.c_[s0.T, s1.T, s2.T]

    # home config
    M = np.array([[1, 0, 0, 3],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # desired end-effector configuration
    T = np.array([[-0.585, -0.811, 0, 0.076],
                  [0.811, -0.585, 0, 2.608],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    thetalist0 = np.array([np.pi/4,
                           np.pi/4,
                           np.pi/4])  # initial guess
    eomg = 0.001  # (0.057 degrees) tolerance of angular velocities
    ev = 0.0001  # (0.1 mm) tolerance of linear velocities


    thetalist, success = IKinSpace(Slist, M, T, thetalist0, eomg, ev)
    assert success
    np.testing.assert_array_almost_equal(thetalist, [0.925198, 0.586225, 0.684273])

