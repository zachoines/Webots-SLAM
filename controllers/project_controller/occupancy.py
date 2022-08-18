from utility import map, bresenham, print_lidar
from matrix_transform import rot_mat
import numpy as np
import matplotlib.pyplot as plt


class OccupancyMap:

    def __init__(self, world_map_size, map_resolution):
        self.world_map_size = world_map_size
        self.map_resolution = map_resolution
        self.shape = (int(self.world_map_size / self.map_resolution), int(self.world_map_size / self.map_resolution))
        self.probability_map = np.zeros(self.shape) + 0.5
        self.probability_map_corrected = None

    def update(self, readings, global_position, global_bearing, printing=False, local_print=False):

        ang, dist = readings
        ox = np.sin(ang) * dist
        oy = np.cos(ang) * dist
        lidar_local = np.array([[x, y] for x, y in zip(ox, oy)])
        lidar_global = self.transform_points_to_frame(global_bearing, global_position, lidar_local)
        lidar_global = np.clip(lidar_global, -self.world_map_size / 2, self.world_map_size / 2)

        if printing:
            if local_print:
                print_lidar((0, 0), lidar_local[:, 0], lidar_local[:, 1])
            else:
                print_lidar(global_position, lidar_global[:, 0], lidar_global[:, 1])

        pmap_tmp = self.generate_occupancy_map(global_position, lidar_global[:, 0], lidar_global[:, 1])
        self.probability_map = self.basic_occ_map_merge(self.probability_map, pmap_tmp) # TODO: Replace this with log-odds representation
        self.probability_map_corrected = np.flip(self.probability_map.T, 0)
        # self.probability_map_corrected = np.flip(self.probability_map,  axis=0)
        # self.probability_map_corrected = self.probability_map.copy()

    def get_map(self):
        return self.probability_map_corrected.copy()

    def generate_occupancy_map(self, robot_pos, ox, oy):
        # Determine bounds of the occupancy map

        ox_min, oy_min = min(ox), min(oy)
        ox_max, oy_max = max(ox), max(oy)
        max_x = self.world_map_size / 2
        max_y = max_x
        max_x = np.floor(max(ox_max, max_x))
        max_y = np.floor(max(oy_max, max_y))
        min_x = min(-max_x, ox_min)
        min_y = min(-max_y, oy_min)
        x_w = int(round(self.world_map_size / self.map_resolution))
        y_w = x_w

        occupancy_map = np.zeros((x_w, y_w)) + 0.5  # default probability for the map is 0.5
        robot_x = int(np.floor(np.round(map(robot_pos[0], min_x, max_x, 0, occupancy_map.shape[0] - 1))))
        robot_y = int(np.floor(np.round(map(robot_pos[1], min_y, max_y, 0, occupancy_map.shape[1] - 1))))

        occupied_x_grid_index = np.clip(np.round((ox - min_x) / self.map_resolution), 0, x_w - 1).astype(int)  #  occupied area x coord
        occupied_y_grid_index = np.clip(np.round((oy - min_y) / self.map_resolution), 0, y_w - 1).astype(int)  #  occupied area y coord

        # occupancy grid computed with bresenham ray casting
        for (x, y) in zip(occupied_x_grid_index, occupied_y_grid_index):
            ray_cast = bresenham((robot_x, robot_y), (x, y))  # line form the lidar to the occupied point
            for occupancy_point in ray_cast:
                if 0 <= occupancy_point[0] < x_w and 0 <= occupancy_point[1] < y_w:
                    occupancy_map[occupancy_point[0]][occupancy_point[1]] = 0.0  # free area 0.0

            self.extend_occupied(occupancy_map, x, y)

        return occupancy_map

    @staticmethod
    def transform_points_to_frame(theta, trans, points):
        rot = rot_mat(theta).T
        trans = np.array(trans)
        transformed = []
        for p in points:
            rotated = rot @ p
            translated = rotated + trans
            transformed.append(translated)
        return np.array(transformed)

    def world_to_grid(self, point):

        min_x, max_x = -self.world_map_size / 2, self.world_map_size / 2
        min_y, max_y = -self.world_map_size / 2, self.world_map_size / 2
        new_x = int(np.floor(np.round(map(point[0], min_x, max_x, 0, self.probability_map.shape[0] - 1))))
        new_y = int(np.floor(np.round(map(point[1], min_y, max_y, 0, self.probability_map.shape[1] - 1))))
        return new_x, new_y

    def grid_to_world(self, point):

        min_x, max_x = -self.world_map_size / 2, self.world_map_size / 2
        min_y, max_y = -self.world_map_size / 2, self.world_map_size / 2

        new_x = map(point[0], 0, self.probability_map.shape[0] - 1, min_x, max_x)
        new_y = map(point[1], 0, self.probability_map.shape[1] - 1, min_y, max_y)

        return new_x, new_y

    @staticmethod
    def extend_occupied(occupancy_map, x, y):
        shape = occupancy_map.shape
        x_next = (x + 1) if (shape[0] - 1) >= (x + 1) else x
        y_next = (y + 1) if (shape[1] - 1) >= (y + 1) else y
        # x_next = x + 1
        # y_next = y + 1
        occupancy_map[x][y] = 1.0
        occupancy_map[x_next][y] = 1.0
        occupancy_map[x][y_next] = 1.0
        occupancy_map[x_next][y_next] = 1.0

    @staticmethod
    def basic_occ_map_merge(map_old, map_new):
        updated = []
        for oldRow, newRow in zip(map_old, map_new):
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

    @staticmethod
    def print_occupancy_map(m):
        shape = np.array(m).shape
        plt.figure(figsize=(10, 10))
        plt.subplot(122)
        plt.imshow(m, cmap="PiYG_r")
        plt.clim(-0.4, 1.4)
        plt.gca().set_xticks(np.arange(-.5, shape[0], 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, shape[1], 1), minor=True)
        plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
        plt.colorbar()
        plt.show()

    def print_path(self, path):
        pmap = self.get_map()
        for x, y in path:
            pmap[int(x)][int(y)] = 2

        self.print_occupancy_map(pmap)