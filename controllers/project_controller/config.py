import math
from matrix_transform import *
from enum import Enum
import numpy as np
ROBOT_STATE = {
    "X": 0,
    "Y": 1,
    "Z": 2,
    "THETA": 3,
    "SIZE": 4
}

CONTROL_SIGNALS = {
    "V": 0,
    "W": 1
}

A_MAX = 6.28  # rad/s
V_MAX = 0.25  # m/s
ACCEL_MAX = 2.00  # m/s^2
L = 52  # mm
D_THRESH = 0.01  # m
A_THRESH = 0.005
OG_RES = 0.02  # occupancy grid resolution
MAP_SIZE = 8
OCC_MAP_UPDATE_RATE = 20
A_STAR_PATH_RECALC_RATE = 20
W_POS_L = [-0.26, 1.59, .12, 1]
R_POS_C = [0.03, 0.0, 0.028, 1]
NUM_LANDMARKS = 1
WHEEL_RADIUS = 0.020
UPDATE_FREQ = 5
LANDMARK_STATE_SIZE = 3


# Noise Covariance
STD_M = 0.001  # measurements
STD_N = [0.001, np.pi / 600]  # control signal
STD_X = [0.001, 0.001, 0.001, np.pi / 900]
STD_L = 0.001