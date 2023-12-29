import numpy as np


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
