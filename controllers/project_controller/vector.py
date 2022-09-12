import numpy as np
import math


def distance(a, b):
    return np.linalg.norm(a - b)


def normalize(v):
    return v / np.sqrt(np.sum(v ** 2))


def vect_unit(v):
    return v[:-1] / v[-1]


def polarToCart(rho, theta):
    return rho*np.cos(theta), rho*np.sin(theta)


# Unit length vector pointing in "look" direction
def heading_vector(phi):

    return np.array([
        np.cos(phi),
        np.sin(phi)
    ])


# Angle from -180 to 180 between two vectors
def signed_angle(a, b):
    unsigned_angle = angle(a, b)
    sign = np.sign(np.cross(a, b))
    return unsigned_angle * sign

# Angle between two vectors
def angle(a, b):
    dem = (np.linalg.norm(a) * np.linalg.norm(b))
    if dem < 1e-15:
        return 0
    try:
        return math.acos(
            np.clip((a @ b) / dem, -1.0, 1.0)
        )
    except:
      return 0


# Angle between the forward vector of our robot (direction_vector) and (X, Y) location of target
# https://math.stackexchange.com/questions/2062021/finding-the-angle-between-a-point-and-another-point-with-an-angle
def angle_to(current_pos, target_pos, direction_vector, degrees=False, signed=False):
    if not signed:
        vector_to = target_pos - current_pos
        theta = angle(direction_vector, vector_to)
        return np.rad2deg(theta) if degrees else theta
    else:
        vector_to = target_pos - current_pos
        theta = signed_angle(vector_to, direction_vector)
        return np.rad2deg(theta) if degrees else theta


