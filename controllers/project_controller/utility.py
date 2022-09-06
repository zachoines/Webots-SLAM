import cv2

from vector import *
import numpy as np
import matplotlib.pyplot as plt


def to_tuple(regular_list):
    return tuple(i for i in regular_list)


def flatten_list(regular_list):
    return [item for sublist in regular_list for item in sublist]


def distance(a, b):
    return np.sum(np.square(a - b))


def print_lidar(rx, ry, x, y):
    plt.figure(figsize=(8, 8))
    plt.plot(
        [x, np.ones(np.size(x)) * rx],
        [y, np.ones(np.size(y)) * ry],
        "ro-",
        scalex=False,
        scaley=False
    )  # lines from 0,0 to the
    plt.axis("equal")
    plt.grid(True)
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
        return x2 - x1, y2 - y1

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
    if swap:
        all_points.reverse()
    return np.array(all_points)


def map(x, in_min, in_max, out_min, out_max):
    mapped = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return mapped


def get_lidar_readings(lidar, ranges=[(-180, 180)]):
    dists, range_max = np.array(lidar.getRangeImage()), lidar.getMaxRange()
    num_points = len(dists)
    angles = np.array([])
    points_in_range = int(num_points / len(ranges))
    for start, end in ranges:
        new_angles = np.linspace(start, end, points_in_range, endpoint=False)
        angles = np.concatenate((angles, np.deg2rad(new_angles)))

    filtered_dists = []
    for dist in dists:
        if dist > range_max:
            filtered_dists.append(range_max)
        else:
            filtered_dists.append(dist)
    return np.array(angles), np.array(filtered_dists)


def stop_robot(lm, rm):
    lm.setVelocity(0.0)
    rm.setVelocity(0.0)


def add_noise(v, loc, sig, shape):
    return v + np.random.normal(loc=loc, scale=sig, size=shape)
