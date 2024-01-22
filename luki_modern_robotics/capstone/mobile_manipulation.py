import os
from collections import namedtuple

import numpy as np
import scipy
from modern_robotics import ScrewTrajectory, Adjoint, MatrixLog6, se3ToVec, JacobianBody, TransInv, FKinBody

# Robot properties
delta_t = 0.01  # seconds

RobotGeometry = namedtuple('RobotGeometry', ['r', 'l', 'omega'])

r = 0.0475  # wheel radius (meters)
l = 0.235  # center to the wheel axis (meters)
omega = 0.15  # half wheel axis (meters)
robot_geometry = RobotGeometry(r, l, omega)


def _get_chassis_config(state):
    """Get the chassis configuration from the state"""
    assert len(state) == 12
    return state[:3]


def _get_angles(state):
    """Get all the joint and wheel angles from the state"""
    assert len(state) == 12
    return state[3:]


def _euler_step(angles, speeds, delta_t):
    """Calculate new angles of joints and wheels for a time step"""
    return angles + speeds * delta_t


def _odometry(q_k, delta_theta):
    """Calculate odometry of mobile base.
    :param q_k: The current chassis config
    :param delta_theta: The wheel angle step
    :return: The new configuration of the mobile base"""

    r = robot_geometry.r
    l = robot_geometry.l
    omega = robot_geometry.omega
    H_0 = 1/r * np.array([[-l - omega, 1, -1],
                          [ l + omega, 1,  1],
                          [ l + omega, 1, -1],
                          [-l - omega, 1,  1],
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
                             (v_bx * np.sin(omega_bz) + v_by * (np.cos(omega_bz) - 1))/omega_bz,
                             (v_by * np.sin(omega_bz) + v_bx * (1 - np.cos(omega_bz)))/omega_bz,
                             ])

    # transforming from body frame to fixed frame
    phi_k = q_k[0]
    transf = np.array([[1, 0, 0],
                        [0, np.cos(phi_k), -np.sin(phi_k)],
                        [0, np.sin(phi_k), np.cos(phi_k)]])
    delta_q = transf @ delta_qb

    # return updated estimate of chassis configuration
    return q_k + delta_q


def NextState(current_configuration,
              wheel_and_joint_controls,
              delta_t: float,
              speed_max: float):
    """
    Calculates the next state of the robot with: euler step for the angles and odometry for the chassis configuration
    :param current_configuration: 12-vector:
        - phi, x, y (chassis configuration)
        - 5 arm configuration angles
        - 4 wheel angles
    :param wheel_and_joint_controls: 9-vector:
        - 5 arm joint speeds: theta_dot
        - 4 wheel speeds: u
    :param delta_t: time step
    :param speed_max: maximum speed of wheels and joints
    """

    assert len(current_configuration) == 12
    assert len(wheel_and_joint_controls) == 9
    # limit speed controls
    controls = np.clip(wheel_and_joint_controls, -speed_max, speed_max)

    # angles
    new_angles = _euler_step(_get_angles(current_configuration), controls, delta_t)
    assert len(new_angles) == 9
    # config
    current_config = _get_chassis_config(current_configuration)
    new_configuration = _odometry(current_config, controls[5:] * delta_t)
    assert len(new_configuration) == 3

    new_state = np.r_[new_configuration, new_angles]
    assert len(new_state) == 12
    return new_state


# Motion Planning #

# Frames:
#  `{s}`: fixed space frame
#  `{b}`: mobile base (body) frame
#  `{0}`: base of arm frame
#  `{e}`: end effector frame
#  `{c}`: cube (manipulated object) frame


def T_se_initial():
    """initial configuration of the end-effector in the reference trajectory"""
    return _T_sb(np.array([0, 0, 0])) @ _T_b_0() @ _M_0e()


def _T_sb(q):
    """configuration of the frame `{b}` of the mobile base, relative to the frame `{s}` on the floor.
    Space frame to chassis frame"""
    phi = q[0]
    x = q[1]
    y = q[2]
    z = 0.0963  # meters is the height of the `{b}` frame above the floor
    se3 = np.array([
        [np.cos(phi), -np.sin(phi), 0, x],
        [np.sin(phi),  np.cos(phi), 0, y],
        [          0,            0, 1, z],
        [          0,            0, 0, 1],
    ])
    return se3


def _T_b_0():
    """The fixed offset from the chassis frame `{b}` to the base frame of the arm `{0}`"""
    return np.array([
        [1, 0, 0, 0.1662],
        [0, 1, 0, 0],
        [0, 0, 1, 0.0026],
        [0, 0, 0, 1]])


def _M_0e():
    """Arm at home configuration (all joint angles zero).
    end-effector frame `{e}` relative to the arm base frame `{0}`"""
    return np.array([
        [1, 0, 0, 0.033],
        [0, 1, 0, 0],
        [0, 0, 1, 0.6546],
        [0, 0, 0, 1]])


def T_sc_initial():
    """Initial configuration of the cube `{c}` relative to the fixed frame `{s}`"""
    return np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0.025],
        [0, 0, 0, 1]])


def T_sc_goal():
    """Goal configuration of the cube `{c}` relative to the fixed frame `{s}`"""
    return np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, -1],
        [0, 0, 1, 0.025],
        [0, 0, 0, 1]])


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


def _grasp_from_cube(cube_config):
    """End effector configuration for grasping cube, relative to cube frame `{c}`"""
    theta = 2

    grasp = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), -0.0215],
        [0, 0, 0, 1]])

    return cube_config @ grasp


def _generate_waypoints(T_se_initial, T_sc_initial, T_sc_goal):
    """Generate all the main positions of the end-effector including gripper state and times required for the step"""
    GRIPPER_OPEN = 0
    GRIPPER_CLOSED = 1
    _ = "unused"

    standoff_1 = _standoff_from_cube(T_sc_initial)
    grip_1 = _grasp_from_cube(T_sc_initial)
    standoff_2 = _standoff_from_cube(T_sc_goal)
    grip_2 = _grasp_from_cube(T_sc_goal)

    np.savetxt("m.csv", T_se_initial, delimiter=",")

    waypoints = []  # (config, gripper, time_in_s)

    waypoints.append((T_se_initial, GRIPPER_OPEN, _))
    waypoints.append((standoff_1, GRIPPER_OPEN, 5))
    waypoints.append((grip_1, GRIPPER_OPEN, 2))
    waypoints.append((grip_1, GRIPPER_CLOSED, 1))
    waypoints.append((standoff_1, GRIPPER_CLOSED, 1))
    waypoints.append((standoff_2, GRIPPER_CLOSED, 5))
    waypoints.append((grip_2, GRIPPER_CLOSED, 2))
    waypoints.append((grip_2, GRIPPER_OPEN, 1))
    waypoints.append((standoff_2, GRIPPER_OPEN, 1))

    return waypoints


def _generate_trajectory(X_from, X_to, gripper, time_in_s):
    """Generate trajectory from one position to another with a given time in seconds.
    Also handle the given gripper state."""
    N = time_in_s / delta_t
    trajectory = ScrewTrajectory(X_from, X_to, time_in_s, N, 3)

    t = []
    # flat list for serialization to csv
    for traj in trajectory:
        r = traj[:-1, :-1]
        p = traj[:-1, -1]
        s = np.concatenate((r.flatten(), p.flatten(), np.array([gripper])))
        t.append(s)

    return t


def TrajectoryGenerator(T_se_initial, T_sc_initial, T_sc_final):
    """Generate trajectory for the pick and place task.
    Intermediate positions (standoff, picking, placing) are calculated with the given positions.
    :param T_se_initial: the initial position of the end effector
    :param T_sc_initial: the initial position of the cube to be picked
    :param T_sc_final: the goal position of the cube
    :return: The trajectory for the given task"""

    waypoints = _generate_waypoints(T_se_initial, T_sc_initial, T_sc_final)
    traj = []
    for i in range(len(waypoints) - 1):
        waypoint_from, _, _ = waypoints[i]
        waypoint_to, gripper, time = waypoints[i + 1]
        tr = _generate_trajectory(waypoint_from,
                                         waypoint_to,
                                         gripper,
                                  time)
        traj.extend(tr)

    return traj


# Control #

M_0_e = np.array([
    [1, 0, 0, 0.033],
    [0, 1, 0, 0],
    [0, 0, 1, 0.6546],
    [0, 0, 0, 1]
])

Blist = np.array([
    [0, 0, 1, 0, 0.033, 0],
    [0, -1, 0, -0.5076, 0, 0],
    [0, -1, 0, -0.3526, 0, 0],
    [0, -1, 0, -0.2176, 0, 0],
    [0, 0, 1, 0, 0, 0],
]).T


def calc_V(X, X_d, X_d_next, k_p, k_i, integral_X_err):  # pylint: disable=too-many-arguments
    """The PD control algorithm for the robot.
    :param X: Current configuration of the end-effector (also written T_se)
    :param X_d: Desired configuration of the end-effector (also written T_se_d)
    :param X_d_next: Desired configuration of the end-effector at next timestep (also written T_se_d_next)
    :param k_p: P gain value
    :param k_i: I gain value"""
    K_p = k_p * np.eye(6)
    K_i = k_i * np.eye(6)

    V_d = se3ToVec(MatrixLog6(TransInv(X_d) @ X_d_next) / delta_t)

    # error twist form X to X_d
    X_err = se3ToVec(MatrixLog6(TransInv(X) @ X_d))

    # Integral (I) part
    integral_X_err += (X_err * delta_t)

    adj_x_inv_x_d = Adjoint(TransInv(X) @ X_d)
    adj_x_inv_x_d_V_d = adj_x_inv_x_d @ V_d

    V = adj_x_inv_x_d_V_d + K_p @ X_err + K_i @ integral_X_err

    return V, integral_X_err, X_err


def calc_J_arm(config):
    theta = config[3:8]
    # J_arm (J_b)
    J_arm = JacobianBody(Blist, theta)
    return J_arm


def calc_J_base(config):
    # J_base
    theta = config[3:8]
    T_0_e = FKinBody(M_0_e, Blist, theta)
    Adj_T_e_b = Adjoint(TransInv(T_0_e))
    r = robot_geometry.r
    l = robot_geometry.l
    omega = robot_geometry.omega
    F = (r / 4) * np.array([
        [-1 / (l + omega), 1 / (l + omega), 1 / (l + omega), -1 / (l + omega)],
        [1, 1, 1, 1],
        [-1, 1, -1, 1]])
    m = F.shape[1]
    F_6 = np.zeros((6, m))
    F_6[2:5, :] = F
    J_base = Adj_T_e_b @ F_6
    return J_base


def calc_J_e(config):
    J_base = calc_J_base(config)
    J_arm = calc_J_arm(config)
    J_e = np.concatenate((J_arm, J_base), axis=1)
    return J_e


def FeedbackControl(X, X_d, X_d_next, config, k_p, k_i, integral_X_err):  # pylint: disable=too-many-arguments
    V, integral_X_err, X_err = calc_V(X, X_d, X_d_next, k_p, k_i, integral_X_err)
    J_e = calc_J_e(config)
    matrix_inv = scipy.linalg.pinv(J_e, atol=0.0001)
    J_e_pinv = matrix_inv
    controls = J_e_pinv @ V

    return controls, integral_X_err, X_err


def to_SE3(waypoint):
    assert len(waypoint) == 12
    r = waypoint[0:9].reshape((3, 3))
    p = waypoint[9:12]

    t = np.c_[r, p]
    t = np.vstack((t, np.array([0, 0, 0, 1])))

    return t


def end_effector_from_config(config):
    T_s_b = _T_sb(config)
    theta = config[3:8]
    T_0_e = FKinBody(M_0_e, Blist, theta)
    T_s_e = T_s_b @ _T_b_0() @ T_0_e
    return T_s_e


def main():  # pylint: disable=too-many-locals
    # First generate a reference trajectory using TrajectoryGenerator and set the initial robot configuration
    # 13-vector: chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper state
    config = np.array([np.pi / 6, -0.1, 0.1, 0, -0.2, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
    theta = config[3:8]
    T_0_e = FKinBody(M_0_e, Blist, theta)
    X = _T_sb(np.array([0, 0, 0])) @ _T_b_0() @ T_0_e
    trajectory = TrajectoryGenerator(X, T_sc_initial(), T_sc_goal())
    all_configurations = np.array([config])

    all_X_err = []

    k_p = 1
    k_i = 0.01
    integral_X_err = np.zeros(6)

    # Loops through the reference trajectory
    for i in range(len(trajectory) - 1):
        # Calculate the control law and generate the wheel and joint controls
        X_d = to_SE3(trajectory[i][:12])
        X_d_next = to_SE3(trajectory[i + 1][:12])

        controls, integral_X_err, X_err = FeedbackControl(X, X_d, X_d_next, config, k_p, k_i, integral_X_err)

        # store every X_err 6-vector, so you can later plot the evolution of the error over time.
        all_X_err.append(X_err)

        gripper = trajectory[i][-1]
        assert gripper in (0.0, 1.0)
        # Use controls, configuration, and timestep to calculate the new configuration
        speed_max = 10
        config = NextState(config[:-1],
                           controls,
                           delta_t,
                           speed_max)
        config = np.append(config, gripper)

        # store configuration for later animation
        all_configurations = np.vstack([all_configurations, config])
        X = end_effector_from_config(config[:-1])

    # Once the program has completed all iterations of the loop:
    # - write out the csv file of configurations: Load the csv file into the CSV Mobile Manipulation youBot scene (Scene 6) to see the results
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(DIR_PATH, "output")
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "output-trajectory.csv"), all_configurations, delimiter=",")
    # - Your program should also generate a file with the log of the X_err 6-vector as a function of time, suitable for plotting
    np.savetxt(os.path.join(output_dir, "x_err.csv"), all_X_err, delimiter=",")


if __name__ == '__main__':
    main()
