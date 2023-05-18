import numpy as np
import matplotlib.pyplot as plt
from modern_robotics import QuinticTimeScaling, ScrewTrajectory, CartesianTrajectory
from sympy import symbols, init_printing, pprint, Eq, linsolve, Poly
from sympy import poly, diff

init_printing(use_unicode=True)


def test_elliptical_path():
    s = np.arange(0, 1, 0.01)

    via_points = [(0, 0),
                  (1.5, 1),
                  (3, 0),
                  (1.5, -1)]

    # 1.
    #x = 3 *(1 - np.cos(2*np.pi * s))
    #y = np.sin(2*np.pi*s)

    # 2.
    x = 1.5 * (1 - np.cos(2 * np.pi * s))
    y = np.sin(2*np.pi*s)

    # 3.
    #x = 1.5 * (1 - np.cos(s))
    #y = np.sin(s)

    # 4.
    #x = np.cos(2 * np.pi * s)
    #y = 1.5 * (1 - np.sin(2*np.pi*s))


    plt.plot(x, y)
    for p in via_points:
        plt.plot(*p, 'ro')

    # plt.show()

def test_5th_order_polynomial():
    a0,a1,a2,a3,a4,a5,t,T = symbols('a0,a1,a2,a3,a4,a5,t,T')
    s = Poly.from_list([a5, a4, a3, a2, a1, a0], t)
    ds = diff(s, t)
    dds = diff(ds, t)
    print()
    pprint(s)
    pprint(ds)
    pprint(dds)

    s_0 = Eq(s(0), 0)
    ds_0 = Eq(ds(0), 0)
    dds_0 = Eq(dds(0), 0)

    s_T =  Eq(s(T), 1)
    ds_T = Eq(ds(T), 0)
    dds_T = Eq(dds(T), 0)

    res = linsolve([s_0, ds_0, dds_0, s_T,  ds_T, dds_T], [a0, a1, a2, a3, a4, a5])
    pprint(res)


def test_quintic_time_scaling():
    T = 5
    t = 3
    s = QuinticTimeScaling(T, t)
    assert s == 0.6825599999999996


x_start = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

x_end = np.array([[0, 0, 1, 1],
                  [1, 0, 0, 2],
                  [0, 1, 0, 3],
                  [0, 0, 0, 1]])
T_f = 10
N = 10
def test_screw_trajectory():
    method = 3
    traj = ScrewTrajectory(x_start, x_end, T_f, N, method)
    assert len(traj) == N
    np.testing.assert_array_almost_equal(traj[-1], x_end)
    np.testing.assert_array_almost_equal(traj[-2],
                                         [[ 0.042292, -0.040573,  0.998281,  0.933132],
                                          [ 0.998281,  0.042292, -0.040573,  1.971986],
                                          [-0.040573,  0.998281,  0.042292,  2.889121],
                                          [ 0.      ,  0.      ,  0.      ,  1.      ]])

def test_cartesian_trajectory():
    method = 5
    traj = CartesianTrajectory(x_start, x_end, T_f, N, method)
    assert len(traj) == N
    np.testing.assert_array_almost_equal(traj[-1], x_end)
    np.testing.assert_array_almost_equal(traj[-2],
                                         [[0.014041, -0.013847, 0.999806, 0.988467],
                                          [0.999806, 0.014041, -0.013847, 1.976934],
                                          [-0.013847, 0.999806, 0.014041, 2.965402],
                                          [0., 0., 0., 1.]])
