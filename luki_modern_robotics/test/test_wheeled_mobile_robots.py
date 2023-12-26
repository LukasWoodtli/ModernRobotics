import numpy as np
from modern_robotics import MatrixExp6, VecTose3, se3ToVec


def h_i_0(r_i, x_i, y_i, gamma_i, beta_i):
    factor = 1/(r_i * np.cos(gamma_i))
    ang = beta_i + gamma_i
    vec = np.array([x_i * np.sin(ang) - y_i * np.cos(ang),
                    np.cos(ang),
                    np.sin(ang)])
    return factor * vec


def _calc_h():
    r = 0.25
    x_1, y_1 = 2, 2
    x_2, y_2 = -2, 2
    x_3, y_3 = -2, -2
    x_4, y_4 = 2, -2
    gamma = 0
    beta_1 = -np.pi / 4
    beta_2 = np.pi / 4
    beta_3 = 3 * np.pi / 4
    beta_4 = -3 * np.pi / 4
    H_0 = np.array([
        h_i_0(r, x_1, y_1, gamma, beta_1),
        h_i_0(r, x_2, y_2, gamma, beta_2),
        h_i_0(r, x_3, y_3, gamma, beta_3),
        h_i_0(r, x_4, y_4, gamma, beta_4),
     ])
    return H_0


def test_calc_h():
    H_0 = _calc_h()
    np.testing.assert_array_almost_equal(H_0, [[-11.313708,   2.828427,  -2.828427],
                                                  [-11.313708,   2.828427,   2.828427],
                                                  [-11.313708,  -2.828427,   2.828427],
                                                  [-11.313708,  -2.828427,  -2.828427]])


def test_omniwheel_robot_1():
    H_0 = _calc_h()
    V_0 = np.array([1, 0, 0])

    u = np.dot(H_0, V_0)
    np.testing.assert_array_almost_equal(u, [-11.313708, -11.313708, -11.313708, -11.313708])


def test_omniwheel_robot_2():
    H_0 = _calc_h()
    V_0 = np.array([1, 2, 3])

    u = np.dot(H_0, V_0)
    np.testing.assert_array_almost_equal(u, [-14.142136,   2.828427,  -8.485281, -25.455844])


def test_control_set():
    v_max = 10 # m/s
    r_min = 2 # m$
    # v = omega * r
    omega = v_max / r_min
    assert omega == 5.0


def test_lie_bracket():  # pylint: disable=too-many-locals
    from sympy import sin, cos, symbols, Matrix, simplify  # pylint: disable=import-outside-toplevel
    from sympy import init_printing, pprint  # pylint: disable=import-outside-toplevel
    init_printing()
    phi, x, y, Theta_L, Theta_R, r, d = symbols("phi x y Theta_L Theta_R r d", real=True)
    q = Matrix([phi, x, y, Theta_L, Theta_R])

    g_1 = Matrix([[-r / (2 * d),
                             (r / 2) * cos(phi),
                             (r / 2) * sin(phi),
                             1,
                             0]]).T

    g_2 = Matrix([[r / (2 * d),
                             (r / 2) * cos(phi),
                             (r / 2) * sin(phi),
                             0,
                             1]]).T

    print()
    res = simplify(g_2.jacobian(q) * g_1 - g_1.jacobian(q) * g_2)
    pprint(res)


def test_four_mecanum_wheel_robot_1():
    r = 1
    l = 3
    omega = 2
    u = np.array([-1.18, 0.68, 0.02, -0.52]).T
    H_0 = 1/r * np.array([[-l - omega, 1, -1],
                          [ l + omega, 1,  1],
                          [ l + omega, 1, -1],
                          [-l - omega, 1,  1],
                          ])
    V_b = np.dot(np.linalg.pinv(H_0), u)
    np.testing.assert_array_almost_equal(V_b, [0.12,-0.25,0.33])


def test_four_mecanum_wheel_robot_2():
    V_b = [0, 0, 0.12,-0.25,0.33, 0]
    m = MatrixExp6(VecTose3(V_b))
    m = se3ToVec(m)
    np.testing.assert_array_almost_equal(m, [0.0,0.0,0.119712,-0.269177,0.314227,0.0])
    np.testing.assert_array_almost_equal(m[2:5], [0.119712,-0.269177,0.314227])
