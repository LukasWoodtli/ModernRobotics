import os

import numpy as np
import pytest
from approvaltests import verify_file, Options
from approvaltests.core import Comparator
from approvaltests.namer import NamerFactory

from ..capstone.mobile_manipulation import Robot, RobotConfiguration, Controller, ControlGains, Task, Scene

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

test_task = Task("test", ControlGains(k_p=2, k_i=0.01), Scene())


class NumpyComparator(Comparator):  # pylint: disable=too-few-public-methods

    def __init__(self):
        pass

    def compare(self, received_path: str, approved_path: str) -> bool:
        try:
            received = np.loadtxt(received_path, delimiter=',')
            approved = np.loadtxt(approved_path, delimiter=',')
            return np.allclose(received, approved)
        except:  # pylint: disable=bare-except
            return False


testdata = [
    ("x-movement", np.array([10, 10, 10, 10])),
    ("y-movement", np.array([-10, 10, -10, 10])),
    ("rotation", np.array([-10, 10, 10, -10]))
]


@pytest.mark.parametrize("name,u", testdata)
def test_NextState(name, u):
    robot = Robot(test_task)
    controller = Controller(robot.robot_geometry,
                            k_p=robot.control_gains.k_p,
                            k_i=robot.control_gains.k_i,
                            delta_t=robot.delta_t)
    steps = 100
    current_config = RobotConfiguration(np.zeros(13))
    theta_dot = np.zeros(5)
    speeds = np.r_[theta_dot, u]
    gripper_state = 0
    current_config.set_gripper(gripper_state)
    all_states = current_config.as_array()
    for _ in range(steps):
        current_config = controller._next_state(current_config, speeds, 5)  # pylint: disable=protected-access
        current_config.set_gripper(gripper_state)
        all_states = np.vstack([all_states, current_config.as_array()])
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

config = RobotConfiguration(np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0]))


def test_calc_V():
    robot = Robot(test_task)
    robot.k_p = 0
    robot.k_i = 0
    controller = Controller(robot_geometry=robot.robot_geometry, k_p=robot.k_p, k_i=robot.k_i, delta_t=robot.delta_t)
    res, X_err = controller._calc_V(X, X_d, X_d_next)  # pylint: disable=protected-access
    expected_V = np.array([0, 0, 0, 21.4, 0, 6.45])
    np.testing.assert_array_almost_equal(res, expected_V)
    np.testing.assert_array_almost_equal(controller.integral_X_err, [0., 0.001709, 0., 0.000795, 0., 0.001067])
    np.testing.assert_array_almost_equal(X_err, [0., 0.170855, 0., 0.079454, 0., 0.106694])


def test_calc_J_e():
    robot = Robot(test_task)
    controller = Controller(robot_geometry=robot.robot_geometry,
                            k_p=robot.control_gains.k_p,
                            k_i=robot.control_gains.k_i,
                            delta_t=robot.delta_t)
    J_e = controller._calc_J_e(config)  # pylint: disable=protected-access
    expected = np.array(
        [[-0.98544973, 0, 0., 0., 0., 0.03039537, -0.03039537, -0.03039537, 0.03039537],
         [0., -1., -1., -1., 0., 0., 0., 0., 0.],
         [0.16996714, 0., 0., 0., 1., -0.00524249, 0.00524249, 0.00524249, -0.00524249],
         [0., -0.24000297, -0.21365806, -0.2176, 0., 0.00201836, 0.00201836, 0.00201836, 0.00201836],
         [0.2206135, 0., 0., 0., 0., -0.01867964, 0.01867964, -0.00507036, 0.00507036],
         [0., -0.28768714, -0.13494244, 0., 0., 0.01170222, 0.01170222, 0.01170222, 0.01170222]])

    np.testing.assert_array_almost_equal(J_e, expected, decimal=3)


def test_calc_FeedbackControl():
    robot = Robot(test_task)
    robot.k_p = 0
    robot.k_i = 0
    controller = Controller(robot_geometry=robot.robot_geometry, k_p=robot.k_p, k_i=robot.k_i, delta_t=robot.delta_t)
    controls, X_err = controller._feedback_control_step(X, X_d, X_d_next, config)  # pylint: disable=protected-access
    expected = np.array([-1.847390e-13, -6.526204e+02, 1.398037e+03, -7.454164e+02,
                         7.707381e-14, 1.571068e+02, 1.571068e+02, 1.571068e+02,
                         1.571068e+02])
    np.testing.assert_array_almost_equal(controls,
                                         expected, decimal=3)
    np.testing.assert_array_almost_equal(controller.integral_X_err,
                                         [0., 0.001709, 0., 0.000795, 0., 0.001067])
    np.testing.assert_array_almost_equal(X_err,
                                         [0., 0.17, 0., 0.08, 0., 0.11], decimal=2)


def test_to_SE3():
    w = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    t = Controller._to_SE3(w)  # pylint: disable=protected-access
    np.testing.assert_array_equal(t, np.array([[1, 2, 3, 10],
                                               [4, 5, 6, 11],
                                               [7, 8, 9, 12],
                                               [0, 0, 0, 1]]))


def test_main():
    robot = Robot(test_task)

    robot.run()

    output_file = os.path.join(os.path.dirname(__file__), '..', 'capstone', 'output', 'test', 'coppelia-sim.csv')

    verify_file(output_file, options=Options().with_comparator(NumpyComparator()))
