from controller import Supervisor
from controller import CameraRecognitionObject

# Local class imports
from pid import PID
from AStar import AStar

# Other contrib imports
import numpy as np
import math
import matplotlib.pyplot as plt


def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# Unit length vector pointing in "look" direction
def heading_vector(phi):
    return np.array([
        np.cos(phi),
        -np.sin(phi)
    ])


# Angle from -180 to 180 between two vectors
def signed_angle(a, b):
    unsigned_angle = angle(a, b)
    sign = np.sign(np.cross(a, b))
    return unsigned_angle * sign


# Angle between two vectors
def angle(a, b):
    dem = (np.linalg.norm(a) * np.linalg.norm(b))

    if dem == 0:
        return 0
    try:
        return math.acos(
            (a @ b) / dem
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


# Euclidean distance
def distance(a, b):
    return np.sum(np.square(a - b))


# Robot's orientation with global north
def get_bearing_in_degrees(compass):
    (x, y, _) = compass.getValues()
    rad = np.arctan2(y, x)
    bearing = (rad - 1.5708) / math.pi * 180.0
    if bearing < 0.0:
        bearing = bearing + 360.0
    return bearing


# NOTE: Use this for a reference location
def robot_state_ground_truth():
    global_position = np.array(gps.getValues()[0:2])
    global_bearing = np.radians(get_bearing_in_degrees(compass))
    return global_position, global_bearing

def sample_route():
    # Note: Make sure to place robot in the bottom right corner of right-hand maze
    return [
        [1.35, -0.7],
        [1.35, 0.0],
        [0.85, 0.0],
        [0.44, -0.7],
        [0.0, -1.0],
        [-0.44, -0.7],
        [-1.13, 0.0],
        [-1.35, 0.0],
        [-1.35, -0.77],
        [-2.83, -0.77],
        [-2.83, -0.74],
        [-2.83, 0.74],
        [-1.44, 0.74],
        [-1.44, 0.0],
        [-0.87, 0.0],
        [-0.44, 0.7],
        [0, 1],
        [0.44, .7],
        [0.9, 0.0]
    ]

# Euclidean distance
def distance(a, b):
    return np.linalg.norm(a - b)


def stop_robot():
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)


def move_robot(v, w, lm, rm):
    v_left = (v - w) * L * 0.5
    v_right = (v + w) * L * 0.5

    lm.setVelocity(v_left)
    rm.setVelocity(v_right)


def get_lidar_readings(l):
    dists, range_max = np.array(l.getRangeImage()), l.getMaxRange()
    num_points = len(dists)
    angles = np.linspace(0.0, np.pi, num_points)
    filtered_dists = []
    for dist in dists:
        if dist > range_max:
            filtered_dists.append(range_max)
        else:
            filtered_dists.append(dist)
    return angles, filtered_dists


def transform_points_to_frame(theta, trans, points):
    rot = rotMat(theta).T
    trans = np.array(trans)
    transformed = []
    for p in points:
        rotated = rot @ p
        translated = rotated + trans
        transformed.append(translated)
    return np.array(transformed)


def printLidar(robot_pos, x, y):
    plt.figure(figsize=(10, 10))
    rx, ry = robot_pos
    plt.plot([
        x,
        np.ones(np.size(x)) * rx],
        [y, np.ones(np.size(y)) * ry],
        "ro-",
        scalex=False,
        scaley=False
    )  # lines from 0,0 to the
    plt.axis("equal")
    plt.grid(True)
    plt.show()



def printOccupancyMap(pmap):
    pmap = np.flip(pmap.T, 0)  # For some reason the axis' are swapped and the pixel rotated 90
    xyres = np.array(pmap).shape
    plt.figure(figsize=(10, 10))
    plt.subplot(122)
    plt.imshow(pmap, cmap="PiYG_r")
    plt.clim(-0.4, 1.4)
    plt.gca().set_xticks(np.arange(-.5, xyres[0], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, xyres[1], 1), minor=True)
    plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
    plt.colorbar()
    plt.show()


def bresenham(current, target):
    """
    Bresenham's Line Algorithm
    wikipedia.org/wiki/Bresenham's_line_algorithm
    https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
    """

    swap = False
    steep = False
    pixel_step = 1
    start, end = current, target

    def delta(t1, t2):
        x1, y1 = t2
        x2, y2 = t1
        return (x2 - x1, y2 - y1)

    def swap_tuple(t1, t2):
        tmp = t1
        t1 = t2
        t2 = tmp
        return t1, t2

    def reverse(t):
        return t[::-1]

    def check_steep(t1, t2):
        diff = delta(t1, t2)
        return abs(diff[1]) > abs(diff[0])

    # If the line is very steep, then rotate it and build up the intersecting points in reverse
    if check_steep(end, start):
        steep = True
        start = reverse(start)
        end = reverse(end)

    # Swap if needed
    if start[0] > end[0]:
        start, end = swap_tuple(start, end)
        swap = True

    # iterate over bounding box generating points between start and end
    all_points = []
    diff = delta(end, start)
    curr_err = np.floor(diff[0] / 2)
    step = pixel_step if start[1] < end[1] else -pixel_step
    y = start[1]
    for x in range(start[0], end[0] + 1):
        all_points.append((y, x) if steep else (x, y))
        curr_err -= abs(diff[1])
        if curr_err < 0:
            y += step
            curr_err += diff[0]

    # Since we swapped points and build the line-up in reverse
    if swap: all_points.reverse()
    return np.array(all_points)



def map(x, in_min, in_max, out_min, out_max):
    mapped = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return mapped


def extend_occupied(occupancy_map, x, y):
    shape = occupancy_map.shape
    x_next = (x + 1) if (shape[0] - 1) >= (x + 1) else x
    y_next = (y + 1) if (shape[1] - 1) >= (y + 1) else y
    occupancy_map[x][y] = 1.0
    occupancy_map[x_next][y] = 1.0
    occupancy_map[x][y_next] = 1.0
    occupancy_map[x_next][y_next] = 1.0

def generate_occupancy_map(robot_pos, ox, oy, xy_resolution, map_size=8.0):
    # Determine bounds of the occupancy map

    ox_min, oy_min = min(ox), min(oy)
    ox_max, oy_max = max(ox), max(oy)
    max_x = map_size / 2
    max_y = max_x
    max_x = np.floor(max(ox_max, max_x))
    max_y = np.floor(max(oy_max, max_y))
    min_x = min(-max_x, ox_min)
    min_y = min(-max_y, oy_min)
    x_w = int(round(map_size / xy_resolution))
    y_w = x_w

    occupancy_map = np.zeros((x_w, y_w)) + 0.5  # default probability for the map is 0.5
    robot_x = int(np.floor(np.round(map(robot_pos[0], min_x, max_x, 0, occupancy_map.shape[0] - 1))))
    robot_y = int(np.floor(np.round(map(robot_pos[1], min_y, max_y, 0, occupancy_map.shape[1] - 1))))

    occupied_x_grid_index = np.clip(np.round((ox - min_x) / xy_resolution), 0, x_w - 1).astype(int)  #  occupied area x coord
    occupied_y_grid_index = np.clip(np.round((oy - min_y) / xy_resolution), 0, y_w - 1).astype(int)  #  occupied area y coord

    # occupancy grid computed with bresenham ray casting
    for (x, y) in zip(occupied_x_grid_index, occupied_y_grid_index):

        ray_cast = bresenham((robot_x, robot_y), (x, y))  # line form the lidar to the occupied point
        for occupancy_point in ray_cast:
            occupancy_map[occupancy_point[0]][occupancy_point[1]] = 0.0  # free area 0.0

        extend_occupied(occupancy_map, x, y)

    return occupancy_map

robot = Supervisor()

# Constants
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0  # ms
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
goal_pos = [0.0, 1.0]  # Hardcoded goal, set to just the exit of the right-hand maze

# Init PIDS of for control signals
vPID = PID(0.0, A_MAX, 0.6, 0.9, 0.02, 0.01)
wPID = PID(0.0, A_MAX, 0.2, 0.05, 0.02, 0.01)

# Init robot and sensors
camera = robot.getDevice('camera')
gps = robot.getDevice("gps")
compass = robot.getDevice("compass")
lidar = robot.getDevice('lidar')
lidar.enable(timestep)
gps.enable(timestep)  # For position
compass.enable(timestep)  # For bearing
camera.enable(timestep)
lidar.enablePointCloud()
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))


if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecofnitionSegmentation()
else:
    print("Camera does not have recognition")


def worldToOCCGrid(point, occ_map, map_size=8):
    min_x, max_x = -map_size / 2, map_size / 2
    min_y, max_y = -map_size / 2, map_size / 2
    new_x = int(np.floor(np.round(map(point[0], min_x, max_x, 0, occ_map.shape[0] - 1))))
    new_y = int(np.floor(np.round(map(point[1], min_y, max_y, 0, occ_map.shape[1] - 1))))
    return new_x, new_y


def OCCGridToWorld(point, occ_map, map_size=8):
    min_x, max_x = -map_size / 2, map_size / 2
    min_y, max_y = -map_size / 2, map_size / 2
    new_x = map(point[0], 0, occ_map.shape[0] - 1, min_x, max_x)
    new_y = map(point[1], 0, occ_map.shape[1] - 1, min_y, max_y)
    return new_x, new_y


def getPath(occ_map, robot_pos, goal_pos):
    # start and goal position
    rx, ry = worldToOCCGrid(robot_pos, occ_map, MAP_SIZE)
    gx, gy = worldToOCCGrid(goal_pos, occ_map, MAP_SIZE)
    a_star = AStar(occ_map)
    path = a_star.get_path(rx, ry, gx, gy)
    path.reverse()
    path_converted = []
    for point in path:
        path_converted.append(OCCGridToWorld(point, occ_map, MAP_SIZE))
    return path_converted, path


def print_path(path, occ_map):
    for x, y in path:
        occ_map[int(x)][int(y)] = 2

    printOccupancyMap(occ_map)


# TODO:: Don't do this... Instead use log odd and inverse sensor model update
def basic_occ_map_merge(pmapOld, pmapNew):
    updated = []
    for oldRow, newRow in zip(pmapOld, pmapNew):
        tmp = []
        for old, new in zip(oldRow, newRow):
            if new == 1.0:
                tmp.append(new)
            elif new == 0.0:
                tmp.append(new)
            else:
                tmp.append(old)
        updated.append(tmp)
    return np.array(updated)


def update_occupancy_map(l, pmap_old, position_cur, orientation_cur):
    #  Visualize robots current surroundings in global frame
    ang, dist = get_lidar_readings(l)
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    lidar_local = np.array([[x, y] for x, y in zip(ox, oy)])
    lidar_global = transform_points_to_frame(orientation_cur, position_cur, lidar_local)
    lidar_global = np.clip(lidar_global, -MAP_SIZE / 2,
                           MAP_SIZE / 2)  # For some reason lidar continues past bounds of arena
    # printLidar((0, 0), lidar_local[:, 0], lidar_local[:, 1]) # Print local view of points
    # printLidar(position_cur, lidar_global[:, 0], lidar_global[:, 1]) # Print global view of points

    pmap_uncorrected = generate_occupancy_map(position_cur, lidar_global[:, 0], lidar_global[:, 1], OG_RES)
    pmap_corrected = pmap_uncorrected
    pmap_new = basic_occ_map_merge(pmap_old, pmap_corrected)
    return pmap_new


def crash_detection(lidar, fov=(45, 135), threshold=0.1):
    ang, dist = get_lidar_readings(lidar)
    for a, d in zip(ang, dist):
        if fov[0] <= np.degrees(a) <= fov[1]:
            if d <= threshold:
                return True
            else:
                return False
    return False


# Main loop
occ_map_shape = (int(MAP_SIZE / OG_RES), int(MAP_SIZE / OG_RES))
pmap = np.zeros(occ_map_shape) + 0.5
num_steps = 0
e_stop = False
while robot.step(timestep) != -1:
    num_steps += 1

    # Init empty occupancy map and robot state
    pos_robot, bearing = robot_state_ground_truth()
    pmap = update_occupancy_map(lidar, pmap, pos_robot, bearing)

    # Now iterate through every point
    path, occ_points = getPath(pmap, pos_robot, goal_pos)
    print_path(occ_points, pmap.copy())

    # Skip the first node in path (it's just the current robots position). If path is real long, use every other point.
    # TODO: Makes point skip dynamic
    path = path[1:] if len(path) > 15 else path[1:-1:5]
    for point in path:

        # if num_steps % A_STAR_PATH_RECALC_RATE == 0:
        #     break

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

        print_path(occ_points, pmap.copy())

        # Keep moving until we have reached destination
        while abs(d_pos) > D_THRESH and robot.step(timestep) != -1:
            num_steps += 1

            # Update occ_map every n steps
            if num_steps % OCC_MAP_UPDATE_RATE == 0:
                pmap = update_occupancy_map(lidar, pmap, pos_robot, bearing)

            # Compute the error
            pos_robot, bearing = robot_state_ground_truth()
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

