import os

import numpy as np
import pytest
from ..capstone.mobile_manipulation import NextState

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

testdata = [
    ("x-movement", np.array([10, 10, 10, 10])),
    ("y-movement", np.array([-10, 10, -10, 10])),
    ("rotation", np.array([-10, 10, 10, -10]))
]


@pytest.mark.parametrize("name,u", testdata)
def test_NextState(name, u):
    delta_t = 0.01
    steps = 100
    current_config = np.zeros(12)
    theta_dot = np.zeros(5)
    speeds = np.r_[theta_dot, u]
    gripper_state = np.ones(0)
    all_states = np.array(np.r_[current_config, gripper_state])
    for _ in range(steps):
        current_config = NextState(current_config, speeds, delta_t, 5)
        current_state = np.r_[current_config, gripper_state]
        all_states = np.vstack([all_states, current_state])
    np.savetxt(os.path.join(DIR_PATH, f"output-{name}.csv"), all_states, delimiter=",")
