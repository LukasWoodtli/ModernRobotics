"""Test Grueblers formula."""

from configuration_space.gruebler import gruebler, gruebler2, gruebler3, dof_of_joint


def test_four_bar_linkage():
    """Four bar linkage"""
    m = 3
    N = 4
    J = 4
    f = (1, ) * 4
    assert len(f) == J

    dof = gruebler2(N, f)
    assert dof == 1

    dof = gruebler(m, N, f)
    assert dof == 1

def test_slider_crank_mechanism_1():
    """Slider-crank mechanism, variant 1"""
    f = (1, ) * 4
    N = 4
    dof = gruebler2(N, f)
    assert dof == 1

def test_slider_crank_mechanism_2():
    """Slider-crank mechanism, variant 2"""
    f = (1, 1, 2)
    N = 3
    dof = gruebler2(N, f)
    assert dof == 1

def test_kR_robot():
    """4-R robot."""
    f = ('r',) * 4
    f = [dof_of_joint(j) for j in f]
    N = 5
    dof = gruebler2(N, f)
    assert dof == 4

def test_five_bar_linkage():
    """5-bar linkage."""
    f = ('r',) * 5
    f = [dof_of_joint(j) for j in f]
    N = 5
    dof = gruebler2(N, f)
    assert dof == 2

def test_stephenson_six_bar_linkage():
    """Test Stephensons mechnism."""
    f = ('r',) * 7
    f = [dof_of_joint(j) for j in f]
    N = 6
    dof = gruebler2(N, f)
    assert dof == 1

def test_watt_six_bar_linkage():
    """Test Watts mechnism."""
    f = ('r',) * 7
    f = [dof_of_joint(j) for j in f]
    N = 6
    dof = gruebler2(N, f)
    assert dof == 1

def test_other():
    """Other test."""
    f = ('r', 'p', 'u', 's')
    f = [dof_of_joint(j) for j in f]
    N = 4
    dof = gruebler3(N, f)
    assert dof == 1

def test_delta_robot():
    """Test for the delta robot"""
    N = 17
    f = ("r", "r", "r", "s", "s", "s", "s") * 3
    f = [dof_of_joint(j) for j in f]
    dof = gruebler3(N, f)
    assert dof == 15

def test_steward_gough_platform():
    """Test for steward-gough platform"""
    N = 14
    f = ("u", "p", "s") * 6
    f = [dof_of_joint(j) for j in f]
    dof = gruebler3(N, f)
    assert dof == 6
