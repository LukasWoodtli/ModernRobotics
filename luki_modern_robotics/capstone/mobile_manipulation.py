import numpy as np
from modern_robotics import ScrewTrajectory


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

    r = 0.0475  # wheel radius (meters)
    l = 0.235  # center to the wheel axis (meters)
    omega = 0.15  # half wheel axis (meters)
    H_0 = 1/r * np.array([[-l - omega, 1, -1],
                          [ l + omega, 1,  1],
                          [ l + omega, 1, -1],
                          [-l - omega, 1,  1],
                          ])
    # calculate twist
    V_b = np.dot(np.linalg.pinv(H_0), delta_theta.T)

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
    return _T_sb(np.array([0, 0, 0])) @ _T_b0() @ _M_0e()


def _T_sb(q):
    """configuration of the frame `{b}` of the mobile base, relative to the frame `{s}` on the floor"""
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


def _T_b0():
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
    theta = 3. / 4 * np.pi
    d = 0.1
    standoff = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), d],
        [0, 0, 0, 1]
    ])

    return cube_conf @ standoff


def _grasp_from_cube(cube_config):
    """End effector configuration for grasping cube, relative to cube frame `{c}`"""
    theta = 3. / 4 * np.pi

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

    waypoints = []  # (config, gripper, time_in_s)

    waypoints.append((T_se_initial, GRIPPER_OPEN, _))
    waypoints.append((standoff_1, GRIPPER_OPEN, 7))
    waypoints.append((grip_1, GRIPPER_OPEN, 5))
    waypoints.append((grip_1, GRIPPER_CLOSED, 2))
    waypoints.append((grip_1, GRIPPER_CLOSED, 2))
    waypoints.append((standoff_1, GRIPPER_CLOSED, 5))
    waypoints.append((standoff_2, GRIPPER_CLOSED, 7))
    waypoints.append((grip_2, GRIPPER_CLOSED, 5))
    waypoints.append((grip_2, GRIPPER_OPEN, 2))
    waypoints.append((grip_2, GRIPPER_OPEN, 2))
    waypoints.append((standoff_2, GRIPPER_OPEN, 5))

    return waypoints


def _generate_trajectory(X_from, X_to, gripper, time_in_s):
    """Generate trajectory from one position to another with a given time in seconds.
    Also handle the given gripper state."""
    dt = 0.01
    N = time_in_s / dt
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
        waypoint_from, gripper, _ = waypoints[i]
        waypoint_to, _, time = waypoints[i + 1]
        traj.extend(_generate_trajectory(waypoint_from,
                                         waypoint_to,
                                         gripper,
                                         time))
    return traj
