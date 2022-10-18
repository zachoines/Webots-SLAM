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

LIDAR_RANGE = 2.5
A_MAX = 6.28  # rad/s
V_MAX = 0.25  # m/s
ACCEL_MAX = 2.00  # m/s^2
L = 52  # m
D_THRESH = 0.05  # m
A_THRESH = 0.005
OG_RES = 0.02  # occupancy grid resolution
MAP_BOUNDS = [[4, -4], [-4, 4]]  # (TL), (BR)
OCC_MAP_UPDATE_RATE = 20
A_STAR_PATH_RECALC_RATE = 20
W_POS_L = [-0.26, 1.59, .12, 1]
R_POS_C = [0.03, 0.0, 0.028, 1]
NUM_LANDMARKS = 4
UPDATE_FREQ = 5
LANDMARK_STATE_SIZE = 3
# https://cyberbotics.com/doc/guide/epuck
WHEEL_RADIUS = 0.020  # 0.020
AXLE_LENGTH = 0.0568  # 0.0568 0.043


# Noise Covariance
STD_M = [0.0001, 0.0001, 0.0001, np.pi / 600]  # measurements
STD_N = [0.001, np.pi / 100]  # control signal
STD_X = [0.0001, 0.0001, 0.0001, np.pi / 600] # robot state
STD_L = 0.001  # landmark state
