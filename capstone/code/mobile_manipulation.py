#!/usr/bin/env python3
import dataclasses
import os
from collections import namedtuple
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from modern_robotics import ScrewTrajectory, Adjoint, MatrixLog6, se3ToVec, JacobianBody, TransInv, FKinBody

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "result")

# Frames:
#  `{s}`: fixed space frame
#  `{b}`: mobile base (body) frame
#  `{0}`: base of arm frame
#  `{e}`: end effector frame
#  `{c}`: cube (manipulated object) frame


class RobotGeometry:
    """Geometry of the robot and related functionality"""
    def __init__(self):
        self.r = 0.0475  # wheel radius (meters)
        self.l = 0.235  # center to the wheel axis (meters)
        self.omega = 0.15  # half wheel axis (meters)

    @staticmethod
    def T_sb(q):
        """configuration of the frame `{b}` of the mobile base, relative to the frame `{s}` on the floor.
        Space frame to chassis frame"""
        phi = q[0]
        x = q[1]
        y = q[2]
        z = 0.0963  # meters: the height of the `{b}` frame above the floor
        se3 = np.array([
            [np.cos(phi), -np.sin(phi), 0, x],
            [np.sin(phi), np.cos(phi), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])
        return se3

    @staticmethod
    def T_b_0():
        """The fixed offset from the chassis frame `{b}` to the base frame of the arm `{0}`"""
        return np.array([
            [1, 0, 0, 0.1662],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0026],
            [0, 0, 0, 1]])

    # home configuration of robot (all joint angles are zero)
    M_0_e = np.array([
        [1, 0, 0, 0.033],
        [0, 1, 0, 0],
        [0, 0, 1, 0.6546],
        [0, 0, 0, 1]
    ])

    # List of screw axes of robot
    Blist = np.array([
        [0, 0, 1, 0, 0.033, 0],
        [0, -1, 0, -0.5076, 0, 0],
        [0, -1, 0, -0.3526, 0, 0],
        [0, -1, 0, -0.2176, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]).T


class RobotConfiguration:
    """Configuration of the robot as 13-vector:
    (chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper state)
    """

    def __init__(self, array: np.ndarray):
        self.configuration = array

    def as_array(self):
        """Get the configuration as a Numpy array for calculations"""
        return self.configuration

    def set_gripper(self, gripper):
        """Set the gripper value (closed: 1, opened: 0)"""
        self.configuration[12] = gripper

    def get_chassis_config(self):
        """Get the chassis configuration (phi, x, y) from the state"""
        assert len(self.configuration) == 13
        return self.configuration[:3]

    def get_theta(self):
        """Get the arm joint angles (J1, ..., J5)"""
        return self.configuration[3:8]

    def get_angles(self):
        """Get all the joint and wheel angles (J1, ..., J5, W1, ..., W4) from the state"""
        assert len(self.configuration) == 13
        return self.configuration[3:12]


class Scene:
    """The default scene with the starting and goal position of the cube"""
    @staticmethod
    def T_sc_initial():
        """Initial configuration of the cube `{c}` relative to the fixed frame `{s}`"""
        return np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0.025],
            [0, 0, 0, 1]])

    @staticmethod
    def T_sc_goal():
        """Goal configuration of the cube `{c}` relative to the fixed frame `{s}`"""
        return np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, -1],
            [0, 0, 1, 0.025],
            [0, 0, 0, 1]])


class SceneNewTask:
    """A second scene with another starting and goal position of the cube"""
    # Initial x = 1, y = 1
    # Goal    x = 1, y = -1
    @staticmethod
    def T_sc_initial():
        """Initial configuration of the cube `{c}` relative to the fixed frame `{s}`"""
        return np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 0.025],
            [0, 0, 0, 1]])

    @staticmethod
    def T_sc_goal():
        """Goal configuration of the cube `{c}` relative to the fixed frame `{s}`"""
        return np.array([
            [0, 1, 0, 1],
            [-1, 0, 0, -1],
            [0, 0, 1, 0.025],
            [0, 0, 0, 1]])


class Planner:  # pylint: disable=too-few-public-methods
    """Plan the trajectory"""

    def __init__(self, T_s_e_initial, scene: Scene, delta_t):
        """Initialize the planner
        :param T_se_initial: the initial position of the end effector
        :param scene: the initial and goal position of the cube to be picked
        """
        self.T_s_e_initial = T_s_e_initial
        self.T_sc_initial = scene.T_sc_initial()
        self.T_sc_goal = scene.T_sc_goal()
        self.delta_t = delta_t

    def trajectory_generator(self):
        """Generate trajectory for the pick and place task.
        Intermediate positions (standoff, picking, placing) are calculated with the given positions.
        :return: The trajectory for the given task"""

        waypoints = self._generate_waypoints()
        traj = []
        for i in range(len(waypoints) - 1):
            waypoint_from, _, _ = waypoints[i]
            waypoint_to, gripper, time = waypoints[i + 1]
            tr = self._generate_trajectory(waypoint_from,
                                      waypoint_to,
                                      gripper,
                                      time)
            traj.extend(tr)

        return traj

    def _generate_waypoints(self):
        """Generate all the main positions of the end-effector including gripper state and times
        required for the step
        :return: The waypoint positions for the trajectory"""
        GRIPPER_OPEN = 0
        GRIPPER_CLOSED = 1
        _ = "unused"

        standoff_1 = self._standoff_from_cube(self.T_sc_initial)
        grip_1 = self._grasp_from_cube(self.T_sc_initial)
        standoff_2 = self._standoff_from_cube(self.T_sc_goal)
        grip_2 = self._grasp_from_cube(self.T_sc_goal)

        # (config, gripper, time_in_s)
        waypoints = [
            (self.T_s_e_initial, GRIPPER_OPEN, _),
            (standoff_1, GRIPPER_OPEN, 5),
            (grip_1, GRIPPER_OPEN, 2),
            (grip_1, GRIPPER_CLOSED, 1),
            (standoff_1, GRIPPER_CLOSED, 1),
            (standoff_2, GRIPPER_CLOSED, 5),
            (grip_2, GRIPPER_CLOSED, 2),
            (grip_2, GRIPPER_OPEN, 1),
            (standoff_2, GRIPPER_OPEN, 1)
        ]

        return waypoints

    def _generate_trajectory(self, X_from, X_to, gripper, time_in_s):
        """Generate part of the trajectory from one position to another with a given time in seconds.
        Also handle the given gripper state.
        :param X_from: starting position of the trajectory segment
        :param X_end: end position of the trajectory segment
        :param gripper: gripper state
        :param time_in_s: time for the trajectory segment
        :return: Calculated trajectory segment
        """
        N = time_in_s / self.delta_t
        trajectory = ScrewTrajectory(X_from, X_to, time_in_s, N, 3)

        t = []
        # flat list for serialization to csv
        for traj in trajectory:
            r = traj[:-1, :-1]
            p = traj[:-1, -1]
            s = np.concatenate((r.flatten(), p.flatten(), np.array([gripper])))
            t.append(s)

        return t

    @staticmethod
    def _standoff_from_cube(cube_conf):
        """End effector standoff configuration, relative to cube frame `{c}`"""
        theta = 2
        d = 0.2
        standoff = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), d],
            [0, 0, 0, 1]
        ])

        return cube_conf @ standoff

    @staticmethod
    def _grasp_from_cube(cube_config):
        """End effector configuration for grasping cube, relative to cube frame `{c}`"""
        theta = 2

        grasp = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), -0.0215],
            [0, 0, 0, 1]])

        return cube_config @ grasp


class Controller:  # pylint: disable=too-few-public-methods
    """The PI control algorithm"""
    def __init__(self, robot_geometry, k_p, k_i, delta_t):
        self.robot_geometry = robot_geometry
        self.k_p = k_p
        self.k_i = k_i
        self.delta_t = delta_t
        self.integral_X_err = np.zeros(6)

    @staticmethod
    def _euler_step(angles, speeds, delta_t):
        """Calculate new angles of joints and wheels for a time step"""
        return angles + speeds * delta_t

    def _next_state(self,
                    current_configuration: RobotConfiguration,
                    wheel_and_joint_controls,
                    speed_max: float) -> RobotConfiguration:
        """
        Calculates the next state of the robot with: euler step for the angles and odometry for the chassis configuration
        :param current_configuration: 12-vector:
            - phi, x, y (chassis configuration)
            - 5 arm configuration angles
            - 4 wheel angles
        :param wheel_and_joint_controls: 9-vector:
            - 5 arm joint speeds: theta_dot
            - 4 wheel speeds: u
        :param speed_max: maximum speed of wheels and joints
        :return: Calculated next configuration of the robot
        """

        assert len(current_configuration.as_array()) == 13
        assert len(wheel_and_joint_controls) == 9
        # limit speed controls
        controls = np.clip(wheel_and_joint_controls, -speed_max, speed_max)

        # angles
        new_angles = self._euler_step(current_configuration.get_angles(), controls, self.delta_t)
        assert len(new_angles) == 9
        # config
        current_config = current_configuration.get_chassis_config()
        new_configuration = self._odometry(current_config, controls[5:] * self.delta_t)
        assert len(new_configuration) == 3

        default_gripper = 0.0
        new_state = np.r_[new_configuration, new_angles, [default_gripper]]
        assert len(new_state) == 13
        return RobotConfiguration(new_state)

    def _odometry(self, q_k, delta_theta):
        """Calculate odometry of mobile base.
        :param q_k: The current chassis config
        :param delta_theta: The wheel angle step
        :return: The new configuration of the mobile base"""

        r = self.robot_geometry.r
        l = self.robot_geometry.l
        omega = self.robot_geometry.omega
        H_0 = 1 / r * np.array([[-l - omega, 1, -1],
                                [l + omega, 1, 1],
                                [l + omega, 1, -1],
                                [-l - omega, 1, 1],
                                ])
        # calculate twist
        V_b = np.dot(scipy.linalg.pinv(H_0, atol=0.0001), delta_theta.T)

        # elements of the twist
        omega_bz = V_b[0]
        v_bx = V_b[1]
        v_by = V_b[2]

        # coordinate changes relative to body frame
        if np.isclose(0.0, omega_bz):
            delta_qb = V_b
        else:
            delta_qb = np.array([omega_bz,
                                 (v_bx * np.sin(omega_bz) + v_by * (np.cos(omega_bz) - 1)) / omega_bz,
                                 (v_by * np.sin(omega_bz) + v_bx * (1 - np.cos(omega_bz))) / omega_bz,
                                 ])

        # transforming from body frame to fixed frame
        phi_k = q_k[0]
        transf = np.array([[1, 0, 0],
                           [0, np.cos(phi_k), -np.sin(phi_k)],
                           [0, np.sin(phi_k), np.cos(phi_k)]])
        delta_q = transf @ delta_qb

        # return updated estimate of chassis configuration
        return q_k + delta_q

    def _calc_J_base(self, config: RobotConfiguration):
        """Calculate Jacobian of the base"""
        # J_base
        theta = config.get_theta()
        T_0_e = FKinBody(self.robot_geometry.M_0_e, self.robot_geometry.Blist, theta)
        Adj_T_e_b = Adjoint(TransInv(T_0_e))
        r = self.robot_geometry.r
        l = self.robot_geometry.l
        omega = self.robot_geometry.omega
        F = (r / 4) * np.array([
            [-1 / (l + omega), 1 / (l + omega), 1 / (l + omega), -1 / (l + omega)],
            [1, 1, 1, 1],
            [-1, 1, -1, 1]])
        m = F.shape[1]
        F_6 = np.zeros((6, m))
        F_6[2:5, :] = F
        J_base = Adj_T_e_b @ F_6
        return J_base

    def _calc_J_arm(self, config: RobotConfiguration):
        """Calculate Jacobian of the arm"""
        theta = config.get_theta()
        # J_arm (J_b)
        J_arm = JacobianBody(self.robot_geometry.Blist, theta)
        return J_arm

    def _calc_J_e(self, config: RobotConfiguration):
        """Calculate Jacobian of the robot"""
        J_base = self._calc_J_base(config)
        J_arm = self._calc_J_arm(config)
        J_e = np.concatenate((J_arm, J_base), axis=1)
        return J_e

    def _calc_V(self, X, X_d, X_d_next):
        """Calculate the twist needed for the PD control algorithm for the robot.
        :param X: Current configuration of the end-effector (also written T_se)
        :param X_d: Desired configuration of the end-effector (also written T_se_d)
        :param X_d_next: Desired configuration of the end-effector at next timestep (also written T_se_d_next)
        :return: The calculated twist for the PD controller"""
        K_p = self.k_p * np.eye(6)
        K_i = self.k_i * np.eye(6)

        V_d = se3ToVec(MatrixLog6(TransInv(X_d) @ X_d_next) / self.delta_t)

        # error twist form X to X_d
        X_err = se3ToVec(MatrixLog6(TransInv(X) @ X_d))

        # Integral (I) part
        self.integral_X_err += (X_err * self.delta_t)

        adj_x_inv_x_d = Adjoint(TransInv(X) @ X_d)
        adj_x_inv_x_d_V_d = adj_x_inv_x_d @ V_d

        V = adj_x_inv_x_d_V_d + K_p @ X_err + K_i @ self.integral_X_err

        return V, X_err

    def _feedback_control_step(self,
                               X,
                               X_d,
                               X_d_next,
                               config: RobotConfiguration):  # pylint: disable=too-many-arguments
        """One step of the control algorithm
        :param X: Current position of end-effector
        :param X_d: Desired position of end-effector
        :param X_d_next: Next desired position of end-effector
        :param config: Configuration of the robot
        :return: tuple of controls and the error
        """
        V, X_err = self._calc_V(X, X_d, X_d_next)
        J_e = self._calc_J_e(config)
        matrix_inv = scipy.linalg.pinv(J_e, atol=0.0001)
        J_e_pinv = matrix_inv
        controls = J_e_pinv @ V

        return controls, X_err

    @staticmethod
    def _to_SE3(waypoint):
        """Converts a trajectory waypoint to a SE3 matrix"""
        assert len(waypoint) == 12
        r = waypoint[0:9].reshape((3, 3))
        p = waypoint[9:12]

        t = np.c_[r, p]
        t = np.vstack((t, np.array([0, 0, 0, 1])))

        return t

    def _end_effector_from_config(self, config: RobotConfiguration):
        """Calculates the end-effector position from the robot configuration
        :param config: the robot configuration
        :return: the end-effector position"""
        # transform from space frame {s} to the body frame {b}
        T_s_b = self.robot_geometry.T_sb(config.get_chassis_config())
        theta = config.get_theta()
        # transform from the base of the robot arm {0} to the end-effector frame {e}
        T_0_e = FKinBody(self.robot_geometry.M_0_e, self.robot_geometry.Blist, theta)
        # finally transform from the space frame {s} to the end-effector frame {e}
        T_s_e = T_s_b @ self.robot_geometry.T_b_0() @ T_0_e
        return T_s_e

    def controller_loop(self, config, trajectory):
        """The complete controller loop.
        :param config: The configuration of the robot
        :param trajectory: The calculated trajectory for the end-effector
        :return: list of all configurations for the task, list of all error"""
        all_configurations = np.array([config.as_array()])
        all_X_err = []
        X = self._end_effector_from_config(config)
        # Loops through the reference trajectory
        for i in range(len(trajectory) - 1):
            # Calculate the control law and generate the wheel and joint controls
            X_d = self._to_SE3(trajectory[i][:12])
            X_d_next = self._to_SE3(trajectory[i + 1][:12])

            controls, X_err = self._feedback_control_step(X, X_d, X_d_next, config)

            # store every X_err 6-vector, so you can later plot the evolution of the error over time.
            all_X_err.append(X_err)

            gripper = trajectory[i][-1]
            assert gripper in (0.0, 1.0)
            # Use controls, configuration, and timestep to calculate the new configuration
            speed_max = 10
            config = self._next_state(config,
                                    controls,
                                    speed_max)
            config.set_gripper(gripper)

            # store configuration for later animation
            all_configurations = np.vstack([all_configurations, config.as_array()])
            X = self._end_effector_from_config(config)
        return all_X_err, all_configurations


class Robot:
    initial_planned_T_s_e = np.array([[0, 0, 1, 0],
                                      [0, 1, 0, 0],
                                      [-1, 0, 0, 0.5],
                                      [0, 0, 0, 1]
                                      ])

    initial_config = RobotConfiguration(np.array([np.pi / 4, -0.5, 0.5, 1, -0.2, 0.2, -1.6, 0, 0, 0, 0, 0, 0]))

    delta_t = 0.01  # seconds
    robot_geometry = RobotGeometry()

    def __init__(self, task):
        """Initialize the robot for the task"""
        self.name = task.name
        self.scene = task.scene
        self.control_gains = task.control_gains
        self.output_base_dir = task.output_base_dir()

    def run(self) -> None:  # pylint: disable=too-many-locals
        """Run the robot to achieve the desired task"""
        config: RobotConfiguration = self.initial_config

        # First generate a reference trajectory using TrajectoryGenerator and set the initial robot configuration
        planner = Planner(self.initial_planned_T_s_e, self.scene, self.delta_t)
        trajectory = planner.trajectory_generator()

        controller = Controller(self.robot_geometry,
                                self.control_gains.k_p,
                                self.control_gains.k_i,
                                self.delta_t
                                )

        all_X_err, all_configurations = controller.controller_loop(config, trajectory)

        self.write_outputs_to_files(all_X_err, all_configurations)

        print("Done.")

    def write_outputs_to_files(self, all_X_err, all_configurations):
        """Write all necessary output files to corresponding folders"""
        print("Generating animation csv file.")
        os.makedirs(self.output_base_dir, exist_ok=True)
        # File with list of configurations for CoppeliaSim
        np.savetxt(os.path.join(self.output_base_dir, "coppelia-sim.csv"), all_configurations, delimiter=",")
        print("Writing error plot data.")
        # File with the log of the X_err 6-vector as a function of time (used for plotting)
        np.savetxt(os.path.join(self.output_base_dir, "x_err.csv"), all_X_err, delimiter=",")

        # Plot error graph
        fig, ax = plt.subplots()
        ax.plot(all_X_err)
        fig.savefig(os.path.join(self.output_base_dir, "x_err.png"))


# The gains for the PI controller
ControlGains = namedtuple('ControlGains', ['k_i', 'k_p'])


@dataclasses.dataclass
class Task:
    """A task for the robot including name, controller gains and starting and goal positions of the cube."""

    name: str  # The name of the task
    control_gains: ControlGains  # The gains for the PI controller
    scene: Union[Scene | SceneNewTask]  # The scene with the cube positions
    base_dir: Optional[str] = None  # The base folder for the output files

    def output_base_dir(self):
        return self.base_dir if self.base_dir else os.path.join(OUTPUT_DIR, self.name)


if __name__ == '__main__':
    tasks = [
        Task("overshoot", ControlGains(k_p=5, k_i=5), Scene()),
        Task("best", ControlGains(k_p=2.0, k_i=0.0), Scene()),
        Task("newTask", ControlGains(k_p=2.0, k_i=0.0), SceneNewTask()),
    ]
    for t in tasks:
        robot = Robot(t)
        robot.run()

# Testsing Code #
# run with `pytest code/mobile_manipulation.py`
else:
    import os
    from approvaltests import verify_file, Options
    from approvaltests.namer import NamerFactory


import pytest
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

test_task = Task("test", ControlGains(k_p=2, k_i=0.01), Scene(), os.path.join(DIR_PATH, "test"))


def get_numpy_comparator():
    from approvaltests.core import Comparator
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
    return NumpyComparator()

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
    output_file = os.path.join(DIR_PATH, "test", f"output-{name}.csv")
    np.savetxt(output_file, all_states, delimiter=",")
    options = NamerFactory.with_parameters(name)
    options = options.with_comparator(get_numpy_comparator())
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
    controller = Controller(robot_geometry=robot.robot_geometry, k_p=robot.k_p, k_i=robot.k_i,
                            delta_t=robot.delta_t)
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
    controller = Controller(robot_geometry=robot.robot_geometry, k_p=robot.k_p, k_i=robot.k_i,
                            delta_t=robot.delta_t)
    controls, X_err = controller._feedback_control_step(X, X_d, X_d_next,
                                                        config)  # pylint: disable=protected-access
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

    output_file = os.path.join(os.path.dirname(__file__), 'test', 'coppelia-sim.csv')

    verify_file(output_file, options=Options().with_comparator(get_numpy_comparator()))
