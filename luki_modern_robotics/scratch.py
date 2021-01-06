"""Scratch file"""
import math
import numpy as np
import robopy.base.transforms as tf
import matplotlib.pyplot as plt
import modern_robotics


print(tf.rot2(0.0))

print(tf.rot2(0.2))

R = tf.rot2(30, 'deg')
print(R)


# First and second column
c0 = R[:, 0]
c1 = R[:, 1]

# Columns are orthogonal to each other
assert np.dot(np.transpose(c0), c1) == 0


# Determinant is 1
assert math.isclose(np.linalg.det(R), 1)

# R^-1 = R^T
inv = np.linalg.inv(R)
transp = np.transpose(R)

assert np.allclose(inv, transp)


# Plotting
#plot_points(R)

fig = plt.figure()
ax = fig.add_subplot(111)
x = [0, 1]
y = [0, 0]
plt.scatter(x, y, s=100, marker='o')
#plt.show()


# Test if modern-robotics library is installed properly

se3mat = [[0.0, 0.0, 0.0, 0.0], [0, 0, -1.5708, 2.3562], [0, 1.5708, 0, 2.3562], [0, 0, 0, 0]]
print(se3mat)

T = modern_robotics.MatrixExp6(se3mat)
print(T)

#assert np.allclose(T, [[1, 0, 0, 0], [0, 0.0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 9]])
