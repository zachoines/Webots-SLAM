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

def bresenham_march(img, p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    # tests if any coordinate is outside the image
    if (
            x1 >= img.shape[0]
            or x2 >= img.shape[0]
            or y1 >= img.shape[1]
            or y2 >= img.shape[1]
    ):  # tests if line is in image, necessary because some part of the line must be inside, it respects the case that the two points are outside
        if not cv2.clipLine((0, 0, *img.shape), p1, p2):
            print("not in region")
            return

    steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # takes left to right
    also_steep = x1 > x2
    if also_steep:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = math.fabs(y2 - y1)
    error = 0.0
    delta_error = 0.0
    # Default if dx is zero
    if dx != 0:
        delta_error = math.fabs(dy / dx)

    y_step = 1 if y1 < y2 else -1

    y = y1
    ret = []
    for x in range(x1, x2):
        p = (y, x) if steep else (x, y)
        if p[0] < img.shape[0] and p[1] < img.shape[1]:
            ret.append((p, img[p]))
        error += delta_error
        if error >= 0.5:
            y += y_step
            error -= 1
    if also_steep:  # because we took the left to right instead
        ret.reverse()
    return np.array(ret)


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


def get_lidar_readings(lidar, ranges=[(-180, 180)], range_max=2.5):
    dists = np.array(lidar.getRangeImage())
    num_points = len(dists)
    angles = np.array([])
    points_in_range = int(num_points / len(ranges))
    for start, end in ranges:
        new_angles = np.linspace(start, end, points_in_range, endpoint=False)
        angles = np.concatenate((angles, np.deg2rad(new_angles)))

    dists = np.clip(dists, 0, range_max)
    inf_indexes = dists >= range_max
    return np.array(angles), dists, inf_indexes


def add_noise(v, loc, sig, shape):
    return v + np.random.normal(loc=loc, scale=sig, size=shape)
