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
    cx, cy, cz = comp.getValues()
    global_bearing = np.arctan2(cx, cy)
    return global_bearing


def get_compass_and_gps_measures(gps, compass):
    global_position = np.array(gps.getValues())
    global_bearing = get_bearing_in_degrees(compass)
    return global_position, global_bearing


# NOTE: Use this for a reference location
def robot_state_ground_truth():
    global_position = robot.getFromDef("e-puck").getPosition()
    rotation = np.array(robot.getFromDef("e-puck").getOrientation()).reshape((3, 3))
    angles, _ = cv2.Rodrigues(rotation)
    global_bearing = np.squeeze(angles)[2]
    return global_position, global_bearing


def detect_landmarks_3D_camera(cam, camera_resolution, camera_fov):
    '''
        Simulates 3d camera.

        Returns camera center-points, distance to center of object,
        and 3D POS of detected object relative to camera.
    '''
    detected_objects = cam.getRecognitionObjects()
    camera_positions = []
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
        camera_positions.append(C_POS_L)

    return camera_positions, detected_objects


def get_measurements(gps, compass, camera):
    # global_position, global_bearing = get_compass_and_gps_measures(gps, compass)
    global_position, global_bearing = robot_state_ground_truth()
    landmark_camera_positions, detected_objects = detect_landmarks_3D_camera(camera, camera_resolution, camera_fov)

    # Add Measurement noise (simulates real world)
    # global_position = add_noise(global_position, 0, STD_M[0:3], len(global_position))
    # global_bearing = add_noise(global_bearing, 0, STD_M[3], None)

    all_g_p_l = []
    all_z = []
    for lp in landmark_camera_positions:
        C_POS_L = np.array(lp + [1])

        # Current Rotation and POS of Robot
        W_POS_R = global_position
        G_ROT_R, _ = cv2.Rodrigues(np.array([0, 0, global_bearing]))
        Identity = np.eye(3)

        # Define Baseline Transforms
        W_T_R = T(G_ROT_R, W_POS_R)
        R_T_C = T(Identity, R_POS_C[:-1])
        W_T_C = W_T_R @ R_T_C

        # Get world and landmark positions
        W_POS_L = W_T_C @ C_POS_L
        R_POS_L = I(W_T_R) @ W_POS_L
        all_z.append(R_POS_L[:3])
        all_g_p_l.append(W_POS_L[:3])

    return all_g_p_l, all_z, global_position, global_bearing, detected_objects


# Camera params
# camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, new_camera_matrix, roi = calibrate_camera()
camera_fov = camera.getFov()
camera_resolution = np.array([camera.getWidth(), camera.getHeight()])

# Other program variables
count = 0
v = .05
omega = -0.1
goal_pos = [0.0, 1.0]  # Hardcoded goal for now
u = np.array([v, omega])
agent = None

def omega_to_wheel_speeds(omega, v):
    wd = omega * AXLE_LENGTH * 0.5
    return (v - wd) / WHEEL_RADIUS, (v + wd) / WHEEL_RADIUS


# Init empty occupancy map and robot state
occ_map = OccupancyMap(MAP_BOUNDS, OG_RES)

while robot.step(timestep) != -1:
    count += 1

    if count == 1:
        _, _, pos_robot, bearing, _ = get_measurements(gps, compass, camera)
        agent = EKF_Agent([*pos_robot, bearing], max_landmarks=NUM_LANDMARKS)
        continue

    readings = get_lidar_readings(lidar)

    # Send out control signals and propagate state
    left_v, right_v = omega_to_wheel_speeds(omega, v)
    leftMotor.setVelocity(left_v)
    rightMotor.setVelocity(right_v)
    # leftMotor.setVelocity(0.0)
    # rightMotor.setVelocity(0.0)

    x_hat_t, Sigma_x_t = agent.propagate(u, dt)

    if count % UPDATE_FREQ == 0:

        # Get new measurement signals
        all_g_p_l, all_z, pos_robot, bearing, detected_objects = get_measurements(gps, compass, camera)

        # Update State
        x_hat_t, Sigma_x_t = agent.update(all_z.copy(), all_g_p_l.copy())

        # Update Occupancy Map
        occ_map.update(readings, x_hat_t[ROBOT_STATE["X"]:ROBOT_STATE["Z"]], x_hat_t[ROBOT_STATE["THETA"]])

        if count % 100 == 0:

            pmap = occ_map.get_map()
            occ_map.print_occupancy_map(pmap)
            cat = "meow"


        # Print out estimated vs actual state
        estimated = "Estimated X : [%8.8f, %8.8f, %8.8f, %8.8f]; " + "".join(["Estimated landmark " + str(i) + ": [%8.8f, %8.8f, %8.8f]; " for i in range(NUM_LANDMARKS)])
        print(estimated % to_tuple(x_hat_t))
        pos_robot, bearing = robot_state_ground_truth()
        _, landmark_global_positions = get_robot_and_global_positions(robot, detected_objects)
        actual = "Actual X : [%8.8f, %8.8f, %8.8f, %8.8f]; " + "".join(["Actual landmark " + str(i) + ": [%8.8f, %8.8f, %8.8f]; " for i in range(len(landmark_global_positions))])
        print(actual % (pos_robot[0], pos_robot[1], pos_robot[2], abs(bearing), *np.array(landmark_global_positions).flatten()))



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


