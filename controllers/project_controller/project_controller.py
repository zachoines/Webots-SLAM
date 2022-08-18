from controller import Supervisor

# Local class imports
from pid import PID
from occupancy import OccupancyMap
from AStar import AStar
from utility import *
from config import *
from matrix_transform import *
from EKM_Agent import EKF_Agent

# Other contrib imports
import math
import numpy as np
import cv2
import sys


robot = Supervisor()
robotNode = robot.getFromDef("e-puck")

# Init PIDS of for control signals
vPID = PID(0.0, A_MAX, 0.6, 0.9, 0.02, 0.01)
wPID = PID(0.0, A_MAX, 0.2, 0.05, 0.02, 0.01)

# Init robot and sensors
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0
camera = robot.getDevice('camera')
gps = robot.getDevice("gps")
compass = robot.getDevice("compass")
lidar = robot.getDevice('lidar')
lidar.enable(timestep)
gps.enable(timestep)  # For position
compass.enable(timestep)  # For bearing
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar.enablePointCloud()
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
robot.step(timestep)  # Make initial step for





def get_path(occ_map, robot_pos, goal_pos, print_map=False):
    # start and goal position
    pmap = occ_map.get_map()
    rx, ry = occ_map.world_to_grid(robot_pos)
    gx, gy = occ_map.world_to_gridz(goal_pos)
    a_star = AStar(pmap)
    path = a_star.get_path(rx, ry, gx, gy)
    path.reverse()
    path_converted = []
    for point in path:
        path_converted.append(occ_map.grid_to_world(point))

    if print_map:
        occ_map.print_path(path_converted)

    return path_converted, path, pmap

def get_robot_and_global_positions(robot, landmarks):

    landmark_robot_positions = []
    landmark_global_positions = []

    for landmark in landmarks:
        landmark_wrt_robot = robot.getFromId(landmark.get_id())
        landmark_global_positions.append(landmark_wrt_robot.getPosition())
        r_l_p = landmark_wrt_robot.getPose(robot.getFromDef("e-puck"))
        r_l_p = np.array(r_l_p).reshape((4, 4))[:, -1]
        landmark_robot_positions.append(r_l_p[:-1])

    return np.array(landmark_robot_positions), np.array(landmark_global_positions)


def detect_landmarks_3D_camera(cam, camera_resolution, camera_fov):
    '''
        Simulates 3d camera.

        Returns camera center-points, distance to center of object,
        and 3D POS of detected object relative to camera.
    '''
    detected_objects = cam.getRecognitionObjects()
    center_points = []
    camera_positions = []
    distances = []
    w, h = camera_resolution
    f = (1 / (2 * math.tan(camera_fov * 0.5) / w))  # Focal length

    for obj in detected_objects:


        # TODO: These values would be derived from a 3d camera.
        u, v = obj.get_position_on_image()
        tmp = obj.get_position()
        dist = np.sqrt(np.sum(np.square(tmp)))

        # https://stackoverflow.com/questions/68638009/compute-3d-point-from-2d-point-on-camera-and-its-distance-from-camera
        X = ((f * dist) / np.sqrt((u - w / 2) ** 2 + (v - h / 2) ** 2 + f ** 2))
        Y = -X / f * (u - w / 2)
        Z = -X / f * (v - h / 2)
        C_POS_L = [X, Y, Z]
        center_points.append([u, v])
        camera_positions.append(C_POS_L)
        distances.append(dist)

    return center_points, camera_positions, distances, detected_objects


def crash_detection(lidar, fov=(45, 135), threshold=0.1):
    '''
    Uses lidar object to detect obstacles closer than threshold within specified FOV
    '''
    ang, dist = get_lidar_readings(lidar)
    for a, d in zip(ang, dist):
        if fov[0] <= np.degrees(a) <= fov[1]:
            if d <= threshold:
                return True
            else:
                return False
    return False


# Robot's orientation with global north
def get_bearing_in_degrees(comp):
    '''
    Uses compass object to get global bearing (relative to Global North) in radians
    '''
    cx, cy, _ = comp.getValues()
    rad = np.arctan2(cy, cx)
    b = (rad - 1.5708) / math.pi * 180.0
    if b < 0.0:
        b = b + 360.0
    return b


# NOTE: Use this for a reference location
def robot_state_ground_truth():
    global_position = np.array(gps.getValues())
    global_bearing = np.radians(get_bearing_in_degrees(compass))
    return global_position, global_bearing

# Init Covariance matrices
Sigma_m = np.array([[STD_M**2, 0], [0, STD_M**2]])
Sigma_n = np.array([[STD_N[0]**2, 0], [0, STD_N[1]**2]])


# Init state vector
landmarks = []

# Camera params
# camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, new_camera_matrix, roi = calibrate_camera()
camera_fov = camera.getFov()
camera_resolution = np.array([camera.getWidth(), camera.getHeight()])

# Other program variables
count = 0
v = .06
omega = 0
goal_pos = [0.0, 1.0]  # Hardcoded goal for now
u = np.array([v, omega])
pos_robot, bearing = robot_state_ground_truth()
agent = EKF_Agent([*pos_robot, bearing], max_landmarks=NUM_LANDMARKS)

# occ_map = OccupancyMap(MAP_SIZE, OG_RES)
# readings = get_lidar_readings(lidar)
# occ_map.update(readings, pos_robot[0:2], bearing)
while robot.step(timestep) != -1:
    leftMotor.setVelocity(v / WHEEL_RADIUS)
    rightMotor.setVelocity(v / WHEEL_RADIUS)
    x_hat_t, Sigma_x_t = agent.propagate(u, Sigma_n, dt)

    if count % UPDATE_FREQ == 0:
        landmark_center_points, landmark_camera_positions, landmark_distances, detected_objects = detect_landmarks_3D_camera(camera, camera_resolution, camera_fov)
        all_z = []
        all_w_pos_l = []

        for lp in landmark_camera_positions:
            C_POS_L = np.array(lp + [1])

            # Current Rotation and POS of Robot
            W_POS_R = robotNode.getPosition()
            G_ROT_R, _ = cv2.Rodrigues(np.array([0, 0, -bearing]))   # G_ROT_R = np.reshape(robotNode.getOrientation(), (3, 3))  /* For reference */
            Identity = np.eye(3)

            # Define Baseline Transforms
            W_T_R = T(G_ROT_R, W_POS_R)
            R_T_C = T(Identity, R_POS_C[:-1])
            W_T_C = W_T_R @ R_T_C

            # For Sanity checks
            # C_POS_L_TEST1 = I(R_T_C) @ I(W_T_R) @ W_POS_L
            # C_POS_L_TEST2 = I(W_T_C) @ W_POS_L
            # W_POS_C_TEST = W_T_R @ R_POS_C

            W_POS_L = W_T_C @ C_POS_L
            R_POS_L = I(W_T_R) @ W_POS_L
            # measurements.append(np.expand_dims(R_POS_L[:2], axis=-1))

        landmark_robot_positions, landmark_global_positions = get_robot_and_global_positions(robot, detected_objects)
        for r_pos_l, w_pos_l in zip(landmark_robot_positions, landmark_global_positions):
            all_z.append(r_pos_l)
            all_w_pos_l.append(w_pos_l)

        x_hat_t, Sigma_x_t = agent.update(all_z, all_w_pos_l)

        G_p_r = robot.getFromDef("e-puck").getPosition()
        G_ori_r = robot.getFromDef("e-puck").getOrientation()
        print("---Estimated---")
        print(x_hat_t[0], x_hat_t[1], x_hat_t[2])
        print("---Actual---")
        print(G_p_r[0], G_p_r[1], bearing)
    count += 1
    # Get Lines at this timestep
    # Currently plotting every timestep (remove or plot less frequently)
    # lines = findLines(np.array(lidar.getRangeImage()), lidar.getMaxRange())
    # pass

    '''
            # Detection via hough line algorithm
            tmpMap = OccupancyMap(MAP_SIZE, OG_RES)
            tmpMap.update(readings, x_hat_t[0:2], x_hat_t[2])
            tmpPmap = tmpMap.get_map() 
            lines = hough_line_detection_image(tmpPmap, plot_lines=True)
            measurements = []
            if len(lines):
                for r, theta in lines:
                    x, y = polarToCart(r, theta)
                    shape = occ_map.shape
                    perpendicular_line = np.array(occ_map.grid_to_world((x + (shape[0] / 2), y + (shape[1] / 2))))
                    measurements.append(np.expand_dims(perpendicular_line, axis=-1))
    '''

'''
occ_map = OccupancyMap(MAP_SIZE, OG_RES)

num_steps = 0
e_stop = False
while robot.step(timestep) != -1:
    num_steps += 1

    # Init empty occupancy map and robot state
    pos_robot, bearing = robot_state_ground_truth()
    readings = get_lidar_readings(lidar)
    occ_map.update(readings, pos_robot, bearing)

    # Now iterate through every point
    path, occ_points, pmap = get_path(occ_map, pos_robot, goal_pos)

    # Skip the first node in path (it's just the current robots position). If path is real long, use every other point.
    # TODO: Makes point skip dynamic
    path = path[1:] if len(path) > 15 else path[1:-1:5]
    for point in path:

        # Forcefully halt and recalculate path if we are about to crash
        # However, don't halt if already halted previously. Give the robot a chance to move first.
        if not e_stop and crash_detection(lidar):
            stop_robot()
            vPID.reset()
            wPID.reset()
            e_stop = True
            break
        else:
            e_stop = False

        # Reset loop variants
        vPID.reset()
        wPID.reset()
        d_pos = np.Infinity
        d_theta = np.Infinity

        # Keep moving until we have reached destination
        while abs(d_pos) > D_THRESH and robot.step(timestep) != -1:
            pos_robot, bearing = robot_state_ground_truth()
            num_steps += 1

            # Update occ_map every n steps
            if num_steps % OCC_MAP_UPDATE_RATE == 0:
                readings = get_lidar_readings(lidar)
                occ_map.update(readings, pos_robot, bearing)
                pmap = occ_map.get_map()

            # Compute the error
            d_pos = distance(np.array(pos_robot), np.array(point))  # distance to target
            d_theta = angle_to(  # angle to target
                np.array(pos_robot),
                np.array(point),
                heading_vector(bearing),
                degrees=False,
                signed=True
            )

            # Compute new control signals
            newLinearVelocity, newAngularVelocity = 0, 0
            if abs(d_theta) < D_THRESH:  # Rotate first, then move forward
                newLinearVelocity = vPID.update(d_pos, dt)
                newAngularVelocity = wPID.update(d_theta, dt)
            else:
                newAngularVelocity = wPID.update(d_theta, dt)

            # Send new commands
            move_robot(
                -newLinearVelocity,
                newAngularVelocity,
                leftMotor,
                rightMotor
            )

    stop_robot()
    vPID.reset()
    wPID.reset()
'''


