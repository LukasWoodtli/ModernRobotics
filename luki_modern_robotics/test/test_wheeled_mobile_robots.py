import numpy as np


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
