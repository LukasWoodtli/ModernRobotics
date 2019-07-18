import unittest
from configuration_space.gruebler import *


class GrueblerfFormulaTest(unittest.TestCase):
    def test_four_bar_linkage_and_slider_crank_mechanism(self):
        m = 3
        N = 4
        J = 4
        f = (1, ) * 4
        assert len(f) == J

        dof = gruebler2(N, f)
        self.assertEqual(1, dof)

        dof = gruebler(m, N, f)
        self.assertEqual(1, dof)


if __name__ == '__main__':
    unittest.main()
