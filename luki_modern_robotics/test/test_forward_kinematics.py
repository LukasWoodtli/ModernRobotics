from modern_robotics import Adjoint, TransInv, FKinSpace, FKinBody
import numpy as np

def test_M():  # pylint: disable=too-many-locals,too-many-statements
    # 1
    M = np.array(
        [[1, 0, 0, np.sqrt(3) + 2],
         [0,  1, 0, 0],
         [0,  0, 1, np.sqrt(3) + 1],
         [0, 0, 0, 1]])

    np.testing.assert_array_almost_equal(M,
                                         [[1., 0.,     0., 3.732050807568877],
                                          [0., 1., 0., 0.],
                                          [0., 0., 1., 2.732050807568877],
                                          [0., 0., 0., 1.]]
                                         )

    # 2
    ## S1
    omega_1 = np.array([0, 0, 1])
    q_1 = np.array([1, 0, 0])
    v_1 = np.cross(-omega_1, q_1)
    S_1 = np.r_[omega_1, v_1]

    np.testing.assert_array_equal(omega_1, [0, 0, 1])
    np.testing.assert_array_equal(v_1, [0, -1, 0])
    np.testing.assert_array_equal(S_1, [0, 0, 1, 0, -1, 0])

    ## S2
    omega_2 = np.array([0, 1, 0])
    q_2 = np.array([1, 0, 0])
    v_2 = np.cross(-omega_2, q_2)
    S_2 = np.r_[omega_2, v_2]

    np.testing.assert_array_equal(S_2, [0, 1, 0, 0, 0, 1])

    ## S3
    omega_3 = np.array([0, 1, 0])
    q_3_x = np.sqrt(3) + 1
    q_3_y = 0
    q_3_z = -1
    q_3 = np.array([q_3_x, q_3_y, q_3_z])
    v_3 = np.cross(-omega_3, q_3)
    S_3 = np.r_[omega_3, v_3]

    np.testing.assert_array_almost_equal(S_3, [0., 1., 0., 1., 0., 2.732051])

    ## S4
    omega_4 = np.array([0, 1, 0])
    q_4_x = np.sqrt(3) + 2
    q_4_y = 0
    q_4_z = np.sqrt(3) - 1
    q_4 = np.array([q_4_x, q_4_y, q_4_z])
    v_4 = np.cross(-omega_4, q_4)
    S_4 = np.r_[omega_4, v_4]

    np.testing.assert_array_almost_equal(S_4, [0.,  1.,  0., -0.732051,  0.,  3.732051])

    # S5
    omega_5 = np.zeros(3)
    v_5 = np.array([0, 0, 1])
    S_5 = np.r_[omega_5, v_5]

    np.testing.assert_array_equal(S_5, [0, 0, 0, 0, 0, 1])
    ## S6
    omega_6 = np.array([0, 0, 1])
    q_6_x = np.sqrt(3) + 2
    q_6_y = 0
    q_6_z = np.sqrt(3) + 1
    q_6 = np.array([q_6_x, q_6_y, q_6_z])
    v_6 = np.cross(-omega_6, q_6)
    S_6 = np.r_[omega_6, v_6]

    np.testing.assert_array_almost_equal(S_6, [0.,  0.,  1.,  0., -3.732051,  0.])

    ## S
    S = np.c_[S_1.T, S_2.T, S_3.T, S_4.T, S_5.T, S_6.T, ]

    np.testing.assert_array_almost_equal(S,
                                         [
                                             [0,  0,        0,         0, 0,         0],
                                             [0,  1,        1,         1, 0,         0],
                                             [1,  0,        0,         0, 0,         1],
                                             [0,  0,        1, -0.732051, 0,         0],
                                             [-1, 0,        0,         0, 0, -3.732051],
                                             [0,  1, 2.732051,  3.732051, 1,         0]
                                             ]
                                         )

    # 3
    ## B1 .. B6
    adjoint_M = Adjoint(TransInv(M))

    B_1 = adjoint_M @ S_1
    np.testing.assert_array_almost_equal(B_1, [0., 0., 1., 0., 2.732051, 0.])

    B_2 = adjoint_M @ S_2
    np.testing.assert_array_almost_equal(B_2, [ 0.,  1.,  0.,  2.732051,  0., -2.732051])

    B_3 = adjoint_M @ S_3
    np.testing.assert_array_almost_equal(B_3, [ 0.,  1.,  0.,  3.732051,  0., -1.])

    B_4 = adjoint_M @ S_4
    np.testing.assert_array_equal(B_4, [0., 1., 0., 2., 0., 0.])

    B_5 = adjoint_M @ S_5
    np.testing.assert_array_equal(B_5, [0., 0., 0., 0., 0., 1.])

    B_6 = adjoint_M @ S_6
    np.testing.assert_array_equal(B_6, [0., 0., 1., 0., 0., 0.])

    ## B
    B = np.c_[B_1.T, B_2.T, B_3.T, B_4.T, B_5.T, B_6.T]
    np.testing.assert_array_almost_equal(B, [
                                             [       0,         0,        0,         0, 0, 0],
                                             [       0,         1,        1,         1, 0, 0],
                                             [       1,         0,        0,         0, 0, 1],
                                             [       0,  2.732051, 3.732051,         2, 0, 0],
                                             [2.732051,         0,        0,         0, 0, 0],
                                             [       0, -2.732051,       -1,         0, 1, 0]
                                             ])

    # 4
    theta = np.array([-np.pi/2.,
                       np.pi/2.,
                       np.pi/3.,
                      -np.pi/4.,
                             1.,
                      np.pi/6.,
                      ])

    T_s = FKinSpace(M, S, theta)
    np.testing.assert_array_almost_equal(T_s,
             [[5.000000e-01, 8.660254e-01, 1.072393e-16, 1.000000e+00],
              [2.241439e-01, -1.294095e-01, -9.659258e-01, -1.897777e+00],
              [-8.365163e-01, 4.829629e-01, -2.588190e-01, -4.508508e+00],
              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])

    # 5
    T_b = FKinBody(M, B, theta)
    np.testing.assert_array_almost_equal(T_b,
             [[5.000000e-01, 8.660254e-01, 1.072393e-16, 1.000000e+00],
              [2.241439e-01, -1.294095e-01, -9.659258e-01, -1.897777e+00],
              [-8.365163e-01, 4.829629e-01, -2.588190e-01, -4.508508e+00],
              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])
