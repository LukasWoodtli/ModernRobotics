"""Test Grueblers formula."""
import unittest

from configuration_space.gruebler import gruebler, gruebler2, gruebler3, dof_of_joint


class GrueblerfFormulaTest(unittest.TestCase):
    """Test cases for Grueblers formula"""

    def test_four_bar_linkage(self):
        """Four bar linkage"""
        m = 3
        N = 4
        J = 4
        f = (1, ) * 4
        assert len(f) == J

        dof = gruebler2(N, f)
        self.assertEqual(1, dof)

        dof = gruebler(m, N, f)
        self.assertEqual(1, dof)

    def test_slider_crank_mechanism_1(self):
        """Slider-crank mechanism, variant 1"""
        f = (1, ) * 4
        N = 4
        self.assertEqual(1, gruebler2(N, f))

    def test_slider_crank_mechanism_2(self):
        """Slider-crank mechanism, variant 2"""
        f = (1, 1, 2)
        N = 3
        self.assertEqual(1, gruebler2(N, f))

    def test_kR_robot(self):
        """4-R robot."""
        f = ('r',) * 4
        f = [dof_of_joint(j) for j in f]
        N = 5
        self.assertEqual(4, gruebler2(N, f))

    def test_five_bar_linkage(self):
        """5-bar linkage."""
        f = ('r',) * 5
        f = [dof_of_joint(j) for j in f]
        N = 5
        self.assertEqual(2, gruebler2(N, f))

    def test_stephenson_six_bar_linkage(self):
        """Test Stephensons mechnism."""
        f = ('r',) * 7
        f = [dof_of_joint(j) for j in f]
        N = 6
        self.assertEqual(1, gruebler2(N, f))

    def test_watt_six_bar_linkage(self):
        """Test Watts mechnism."""
        f = ('r',) * 7
        f = [dof_of_joint(j) for j in f]
        N = 6
        self.assertEqual(1, gruebler2(N, f))

    def test_other(self):
        """Other test."""
        f = ('r', 'p', 'u', 's')
        f = [dof_of_joint(j) for j in f]
        N = 4
        self.assertEqual(1, gruebler3(N, f))

    def test_delta_robot(self):
        N = 17
        f = ("r", "r", "r", "s", "s", "s", "s") * 3
        f = [dof_of_joint(j) for j in f]
        self.assertEqual(15, gruebler3(N, f))

    def test_steward_gough_platform(self):
        N = 14
        f = ("u", "p", "s") * 6
        f = [dof_of_joint(j) for j in f]
        self.assertEqual(6, gruebler3(N, f))


if __name__ == '__main__':
    unittest.main()
