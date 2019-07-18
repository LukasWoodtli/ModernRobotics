"""Grueblers formula. See MR p. 15f"""


def _gruebler(m, N, J, fs):
    """
    Calculate dof of a robot using Grueblers formula.
    :param m: dof of a rigid body (planar: m = 3, spacial: m = 6)
    :param N: number of links (including ground)
    :param J: number of joints
    :param fs: list of dof provided by each joint
    """
    assert m == 3 or m == 6
    assert J == len(fs)
    return m * (N - 1 - J) + sum(fs)


def gruebler(m, N, fs):
    """See _gruebler for argument description."""
    return _gruebler(m, N, len(fs), fs)


def gruebler2(N, fs):
    RIGID_BODY_DOF_2D = 3
    return gruebler(RIGID_BODY_DOF_2D, N, fs)


def gruebler3(N, fs):
    RIGID_BODY_DOF_3D = 6
    return gruebler(RIGID_BODY_DOF_3D, N, fs)


def dof_of_joint(joint_type):
    dofs = {
        ["revolute", "r", "scharnier"]: 1,
        ["prismatic", "p", "schubgelenk"]: 1,
        ["cylindrical", "c", "drehchubgelenk"]: 2,
        ["universal", "u", "kardangelenk"]: 2,
        ["spherical", "s", "kugelgelenk"]: 3
    }

    for k, v in dofs:
        if joint_type.lower() in k:
            return v

    raise RuntimeError(f"Can't find joint type: {joint_type}")