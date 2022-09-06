from utility import *
import numpy as np
import matplotlib.pyplot as plt


class OccupancyMap:

    def __init__(self, world_map_bounds, map_resolution):
        self.world_map_bounds = world_map_bounds
        TL, BR = world_map_bounds
        (y1, x1), (y2, x2) = TL, BR
        self.world_map_size = [abs(y2-y1), abs(x2-x1)]
        self.map_resolution = map_resolution
        self.oc_shape = (int(self.world_map_size[1] / self.map_resolution), int(self.world_map_size[0] / self.map_resolution))

        # Inverse sensor model
        # TODO: Determine a smarter way to determine probs here.
        self.p_hit = 0.8
        self.p_miss = 0.2
        self.p_unknown = 0.5

        # Occupancy and Log odds Maps
        self.prior = self.l(np.zeros(self.oc_shape) + self.p_unknown)
        self.log_odds = np.zeros(self.oc_shape)
        self.occ_map = np.zeros(self.oc_shape) + self.p_unknown

    def l(self, p_x):
        '''
        return log odds of P(x)
        '''
        return np.log(
            p_x / (1 - p_x)
        )

    def p(self, log_odds):
        '''
        Retrieve P(x) from Log odds l(P(x))
        '''
        return 1.0 - (1.0 / (1.0 + np.exp(log_odds)))

    def update_log_odds_occ(self, new_map):
        '''
        Update logs odds representation of occ map
        '''
        self.log_odds = self.l(new_map) + self.log_odds - self.prior

    def update(self, readings, global_position, global_bearing, printing=False):
        angles, dist = readings
        dist = np.array(dist)
        ox = np.cos(angles) * dist
        oy = np.sin(angles) * dist

        lidar_local = np.array([[x, y] for x, y in zip(ox, oy)])
        lidar_global = self.transform_points_to_frame(global_bearing, global_position, lidar_local)

        if printing:
            print_lidar(0, 0, oy, ox)

        occ_map_tmp = self.generate_occupancy_map(global_position[0], global_position[1], lidar_global[:, 0], lidar_global[:, 1])
        # self.occ_map = self.basic_occ_map_merge(self.occ_map, occ_map_tmp)
        self.update_log_odds_occ(occ_map_tmp)

    def generate_occupancy_map(self, robot_x, robot_y, ox, oy):

        min_y, min_x = np.min(oy), np.min(ox)
        # occupied_y_grid_index = np.clip(oy, min(self.world_map_bounds[0][0], self.world_map_bounds[1][0]), max(self.world_map_bounds[0][0], self.world_map_bounds[1][0]))
        # occupied_x_grid_index = np.clip(ox, min(self.world_map_bounds[0][1], self.world_map_bounds[1][1]), max(self.world_map_bounds[0][1], self.world_map_bounds[1][1]))
        occupied_y_grid_index = np.clip(np.round((oy - min_y) / self.map_resolution), 0, self.oc_shape[1] - 1).astype(int)
        occupied_x_grid_index = np.clip(np.round((ox - min_x) / self.map_resolution), 0, self.oc_shape[0] - 1).astype(int)

        occupancy_map = np.zeros(self.oc_shape) + self.p_unknown
        robot_y = np.clip(np.round(map(robot_y, self.world_map_bounds[0][0], self.world_map_bounds[1][0], 0, occupancy_map.shape[1] - 1)), 0, self.oc_shape[1] - 1).astype(int)
        robot_x = np.clip(np.round(map(robot_x, self.world_map_bounds[0][1], self.world_map_bounds[1][1], 0, occupancy_map.shape[0] - 1)), 0, self.oc_shape[1] - 1).astype(int)


        # occupancy grid computed with bresenham ray casting
        for (y, x) in zip(occupied_y_grid_index, occupied_x_grid_index):
            # ray_cast = bresenham((robot_y, robot_x), (y, x))  # line form the lidar to the occupied point
            # for ray_y, ray_x in ray_cast:
            #     if 0 <= ray_x < self.oc_shape[1] and 0 <= ray_y < self.oc_shape[0]:
            #         occupancy_map[ray_y][ray_x] = 0.0  # free area 0.0

            cv2.line(occupancy_map, (robot_x, robot_y), (x, y), (0, self.p_miss, 0))

            self.extend_occupied(occupancy_map, y, x)

        return occupancy_map

    def rotMat(self, theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s, c]])

    def transform_points_to_frame(self, theta, trans, points):
        rot = self.rotMat(theta)
        return np.array([
            (rot.T @ (pt - trans))
            for pt in points
        ])

    def get_map(self):
        return self.p(self.log_odds)

    def world_to_grid(self, point):
        min_x, max_x = -self.world_map_size / 2, self.world_map_size / 2
        min_y, max_y = -self.world_map_size / 2, self.world_map_size / 2
        new_x = int(np.floor(np.round(map(point[0], min_x, max_x, 0, self.occ_map.shape[0] - 1))))
        new_y = int(np.floor(np.round(map(point[1], min_y, max_y, 0, self.occ_map.shape[1] - 1))))
        return new_x, new_y

    def grid_to_world(self, point):
        min_x, max_x = -self.world_map_size / 2, self.world_map_size / 2
        min_y, max_y = -self.world_map_size / 2, self.world_map_size / 2
        new_x = map(point[0], 0, self.occ_map.shape[0] - 1, min_x, max_x)
        new_y = map(point[1], 0, self.occ_map.shape[1] - 1, min_y, max_y)
        return new_x, new_y

    def extend_occupied(self, occupancy_map, y, x):
        shape = occupancy_map.shape
        x_next = x + 1 if (shape[0] - 1) >= (x + 1) else x
        y_next = y + 1 if (shape[1] - 1) >= (y + 1) else y

        # TODO::Todo look into ways to inprove this based on properties of the inverse sensor model
        occupancy_map[y][x] = self.p_hit
        occupancy_map[y_next][x] = self.p_hit
        occupancy_map[y][x_next] = self.p_hit
        occupancy_map[y_next][x_next] = self.p_hit

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

    def print_occupancy_map(self, m):
        # cv2.imshow('graycsale image', m)
        m = np.array(m)
        shape = m.shape
        plt.figure(figsize=(2 * self.world_map_size[1], 2 * self.world_map_size[0]))
        plt.subplot(111)
        plt.imshow(m, cmap="PiYG_r")
        plt.clim(-0.4, 1.4)

        plt.gca().set_xticks(np.arange(start=0, stop=shape[0], step=1), minor=True)
        plt.gca().set_yticks(np.arange(start=0, stop=shape[1], step=1), minor=True)

        plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
        plt.colorbar()
        plt.show()

    def print_path(self, path):
        pmap = self.get_map()
        for x, y in path:
            pmap[int(x)][int(y)] = 2

        self.print_occupancy_map(pmap)