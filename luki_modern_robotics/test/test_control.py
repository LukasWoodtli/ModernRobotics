import numpy as np


def test_characteristic_equation_1():
    # 4x'' + 8x' + 12x = 0
    # => x'' + 2x' + 3x = 0
    # => x'' = -2x' -3x
    arr = np.array([[0, 1],
              [-3, -2]])

    poly = np.poly(arr)
    # 1s^2 + 2s + 3 = 0
    np.testing.assert_array_almost_equal(poly, [1, 2, 3])


def test_characteristic_equation_2():
    arr = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [-3, -2, -1]])

    poly = np.poly(arr)
    # 1s^3 + 1s^2 + 2s + 3 = 0
    np.testing.assert_array_almost_equal(poly, [1, 1, 2, 3])
    roots = np.roots(poly)
    stable = all(comp.real < 0 for comp in roots)
    assert not stable


def test_calc_zeta():
    # Characteristic polynomial
    # s^2 + 2s + 2 = 0
    # s^2 + 2 * zeta * omega_n * s + omega_n^2 = 0
    _omega_n = np.sqrt(2.)
    zeta = 1/np.sqrt(2.)
    assert zeta == 0.7071067811865475


def test_calc_w_d():
    # x'' + 3x'' + 9x = 0
    # x'' + 2 * zeta * omega_n * x' + omega_n^2 * x = 0
    # omega_d = omega_n * srt(1 - zeta^2)
    omega_n = 3.
    # zeta: 3 = 2 * zeta * omega_n
    zeta = 3/(2 * omega_n)
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    assert omega_d == 2.598076211353316


def test_calc_kp():
    # s^2 + K_p*s + K_i = 0
    # s^2 + 2 * zeta * omega_n * s + omega_n^2 = 0
    # critical damping:
    zeta = 1
    K_i = 10
    omega_n = np.sqrt(K_i)
    K_p = 2 * zeta * omega_n
    assert K_p == 6.324555320336759


def test_ki_max():
    # s^3 + (b + K_d)/M * s^2 + K_p/M * s + K_i/M = 0
    M = 1
    b = 2
    K_d = 3
    K_p = 4
    K_i_max = ((b + K_d) * K_p)/M
    assert K_i_max == 20.0


def test_kd():
    # s = -4
    # M = 1
    # b = 2
    # (b + K_d)/M * s^2 = 0
    # (2 + K_d)/1 * 16 = 0

    K_d = 1
    assert K_d == 1
