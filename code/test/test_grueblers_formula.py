import unittest

from configuration_space.gruebler import *


class GrueblerfFormulaTest(unittest.TestCase):
    def test_four_bar_linkage(self):
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
        f = (1, ) * 4
        N = 4
        self.assertEqual(1, gruebler2(N, f))

    def test_slider_crank_mechanism_2(self):
        f = (1, 1, 2)
        N = 3
        self.assertEqual(1, gruebler2(N, f))

    def test_kR_robot(self):
        f = ('r',) * 4
        f = [dof_of_joint(j) for j in f]
        N = 5
        self.assertEqual(4, gruebler2(N, f))

    def test_five_bar_linkage(self):
        f = ('r',) * 5
        f = [dof_of_joint(j) for j in f]
        N = 5
        self.assertEqual(2, gruebler2(N, f))

    def test_stepenson_six_bar_linkage(self):
        f = ('r',) * 7
        f = [dof_of_joint(j) for j in f]
        N = 6
        self.assertEqual(1, gruebler2(N, f))

    def test_watt_six_bar_linkage(self):
        f = ('r',) * 7
        f = [dof_of_joint(j) for j in f]
        N = 6
        self.assertEqual(1, gruebler2(N, f))

if __name__ == '__main__':
    unittest.main()
