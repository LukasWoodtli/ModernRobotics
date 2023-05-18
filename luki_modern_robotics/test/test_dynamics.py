import numpy as np
from modern_robotics import InverseDynamics, MassMatrix, VelQuadraticForces, GravityForces, EndEffectorForces, \
    ForwardDynamics

from luki_modern_robotics.test import ur5_parameter  # pylint: disable=import-error


thetalist = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2 ,2*np.pi/3])
dthetalist = np.full(shape=6, fill_value=0.2)
ddthetalist = np.full(shape=6, fill_value=0.1)
g = np.array([0, 0, -9.81]).T
Ftip = np.full(shape=6, fill_value=0.1)
Mlist = ur5_parameter.Mlist
Glist = ur5_parameter.Glist
Slist = ur5_parameter.Slist

def test_ur5():
    forces_torques = InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, \
                    Glist, Slist)
    expected = [0.0127670527, -41.1476981, -3.78093382,  0.0323099099, 0.0369529049,  0.103367459]
    np.testing.assert_array_almost_equal(forces_torques, expected)


def test_mass_matrix():
    M = MassMatrix(thetalist, Mlist, Glist, Slist)
    expected = [
        [2.197766459720294, 0.2722821358005576, 0.06803253136305762, -0.006481516748909031, 0.1702202985913252, -0.012117316366733508 ],
        [0.2722821358005576, 3.553693021743094, 1.3104072975885415, 0.2403370450313314, -0.007225045065805527, 1.963311520010452e-18 ],
        [0.06803253136305762, 1.3104072975885415, 0.8372485118439893, 0.24763800734266944, -0.007225045065805527, 2.0409385261450496e-18 ],
        [- 0.006481516748909032, 0.2403370450313315, 0.24763800734266952, 0.25367945451609997, -0.007225045065805527, 1.902530704689376e-18 ],
        [0.1702202985913252, -0.007225045065805527, -0.007225045065805526, -0.007225045065805526, 0.24072785485905, 0.0 ],
        [- 0.01211731636673351, 1.9633115200104518e-18, 2.0409385261450496e-18, 1.902530704689376e-18, 0.0, 0.0171364731454 ]
    ]

    np.testing.assert_array_almost_equal(M, expected)

def test_vel_quadratic_forces():
    c = VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
    expected = [-0.11745 , -0.010711,  0.031651, -0.014773,  0.023386,  0.002866]
    np.testing.assert_array_almost_equal(c, expected)


def test_gravity_forces():
    grav = GravityForces(thetalist, g, Mlist, Glist, Slist)
    expected = [ 5.47883360e-16, -4.15967263e+01, -3.93590583e+00,  1.23367683e-01,  5.63470848e-18,  4.89532676e-18]
    np.testing.assert_array_almost_equal(grav, expected)


def test_end_effector_forces():
    F_tip = EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)
    expected = [-0.13875364, -0.0772107, -0.12228876, -0.14907993, -0.02536015,  0.1]
    np.testing.assert_array_almost_equal(F_tip, expected)


def test_forward_dynamics():
    taulist = np.array([0.0128,
                        -41.1477,
                        -3.7809,
                        0.0323,
                        0.0370,
                        0.1034])
    dtau = ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist)
    expected = [0.10001202, 0.09994735, 0.10016669, 0.0998537, 0.10018617, 0.10190745]
    np.testing.assert_array_almost_equal(dtau, expected)
