import os

import numpy as np
import pytest
from approvaltests import verify_file, Options
from approvaltests.core import Comparator
from approvaltests.namer import NamerFactory

from ..capstone.mobile_manipulation import Robot

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class NumpyComparator(Comparator):  # pylint: disable=too-few-public-methods

    def __init__(self):
        pass

    def compare(self, received_path: str, approved_path: str) -> bool:
        received = np.loadtxt(received_path, delimiter=',')
        approved = np.loadtxt(approved_path, delimiter=',')
        return np.allclose(received, approved)


testdata = [
    ("x-movement", np.array([10, 10, 10, 10])),
    ("y-movement", np.array([-10, 10, -10, 10])),
    ("rotation", np.array([-10, 10, 10, -10]))
]


@pytest.mark.parametrize("name,u", testdata)
def test_NextState(name, u):
    robot = Robot()
    steps = 100
    current_config = np.zeros(12)
    theta_dot = np.zeros(5)
    speeds = np.r_[theta_dot, u]
    gripper_state = np.ones(0)
    all_states = np.array(np.r_[current_config, gripper_state])
    for _ in range(steps):
        current_config = robot.NextState(current_config, speeds, 5)
        current_state = np.r_[current_config, gripper_state]
        all_states = np.vstack([all_states, current_state])
    output_file = os.path.join(DIR_PATH, f"output-{name}.csv")
    np.savetxt(output_file, all_states, delimiter=",")
    options = NamerFactory.with_parameters(name)
    options = options.with_comparator(NumpyComparator())
    verify_file(output_file, options=options)


X = np.array([
    [0.170, 0, 0.985, 0.387],
    [0, 1, 0, 0],
    [-0.985, 0, 0.170, 0.570],
    [0, 0, 0, 1],
])
X_d = np.array([
    [0, 0, 1, 0.5],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
])
X_d_next = np.array([
    [0, 0, 1, 0.6],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.3],
    [0, 0, 0, 1]

])

config = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0])


def test_calc_V():
    robot = Robot()
    robot.k_p = 0
    robot.k_i = 0
    res, X_err = robot.calc_V(X, X_d, X_d_next)
    expected_V = np.array([0, 0, 0, 21.4, 0, 6.45])
    np.testing.assert_array_almost_equal(res, expected_V)
    np.testing.assert_array_almost_equal(robot.integral_X_err, [0., 0.001709, 0., 0.000795, 0., 0.001067])
    np.testing.assert_array_almost_equal(X_err, [0., 0.170855, 0., 0.079454, 0., 0.106694])


def test_calc_J_e():
    robot = Robot()
    J_e = robot.calc_J_e(config)
    expected = np.array(
        [[-0.98544973, 0, 0., 0., 0., 0.03039537, -0.03039537, -0.03039537, 0.03039537],
         [0., -1., -1., -1., 0., 0., 0., 0., 0.],
         [0.16996714, 0., 0., 0., 1., -0.00524249, 0.00524249, 0.00524249, -0.00524249],
         [0., -0.24000297, -0.21365806, -0.2176, 0., 0.00201836, 0.00201836, 0.00201836, 0.00201836],
         [0.2206135, 0., 0., 0., 0., -0.01867964, 0.01867964, -0.00507036, 0.00507036],
         [0., -0.28768714, -0.13494244, 0., 0., 0.01170222, 0.01170222, 0.01170222, 0.01170222]])

    np.testing.assert_array_almost_equal(J_e, expected, decimal=3)


def test_calc_FeedbackControl():
    robot = Robot()
    robot.k_p = 0
    robot.k_i = 0
    controls, X_err = robot.FeedbackControl(X, X_d, X_d_next, config)
    expected = np.array([-1.847390e-13, -6.526204e+02, 1.398037e+03, -7.454164e+02,
                         7.707381e-14, 1.571068e+02, 1.571068e+02, 1.571068e+02,
                         1.571068e+02])
    np.testing.assert_array_almost_equal(controls,
                                         expected, decimal=3)
    np.testing.assert_array_almost_equal(robot.integral_X_err,
                                         [0., 0.001709, 0., 0.000795, 0., 0.001067])
    np.testing.assert_array_almost_equal(X_err,
                                         [0., 0.17, 0., 0.08, 0., 0.11], decimal=2)


def test_to_SE3():
    w = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    t = Robot.to_SE3(w)
    np.testing.assert_array_equal(t, np.array([[1, 2, 3, 10],
                                               [4, 5, 6, 11],
                                               [7, 8, 9, 12],
                                               [0, 0, 0, 1]]))


def test_main():
    robot = Robot()
    robot.main()

    output_file = os.path.join(os.path.dirname(__file__), '..', 'capstone', 'output', 'output-trajectory.csv')

    verify_file(output_file, options=Options().with_comparator(NumpyComparator()))
