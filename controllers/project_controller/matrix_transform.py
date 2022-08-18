import numpy as np
import math


def intrinsic_camera_matrix(pp, fov):
	f = 1 / (2 * math.tan(fov * 0.5) / 1080)
	# fx = 1211.6447914735534
	return np.array([
		[f, 0, pp[0]],
		[0, f, pp[1]],
		[0, 0, 1]
	])

def rot_mat(theta):
	c, s = math.cos(theta), math.sin(theta)
	return np.array([[c, -s], [s, c]])


def rel_to_global(rel_x, rel_y, x_r, y_r, r_theta):
	c = rot_mat(r_theta)
	rot = c@np.array([[rel_x], [rel_y]])
	rot[0, 0] += x_r
	rot[1, 0] += y_r
	return rot


def invert_homogenous_transformation(h):
	r = h[0:3, 0:3]
	t = h[:, -1][0:3]
	inverted = np.eye(4)
	inverted[0:3, 0:3] = r.T
	tmp = (-r.T @ t).tolist() + [1]
	inverted[:, -1] = tmp
	return inverted


def I(h):
	return invert_homogenous_transformation(h)


def convert_to_homogenous_transformation(r, t):
	M = np.eye(4)
	M[0:3, 0:3] = r
	M[0:3, -1] = t
	return M


def T(r, t):
	return convert_to_homogenous_transformation(r, t)
