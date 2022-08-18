import math
import numpy as np
from utility import bresenham

class AStar:

    def __init__(self, occ_map, goal_threshold=5, robot_radius=3):
        x, y = occ_map.shape
        self.robot_radius = robot_radius
        self.goal_threshold = goal_threshold
        self.min_x, self.max_x = 0, x
        self.min_y, self.max_y = 0, y
        self.motion = self.movement_pattern()
        self.x_width, self.y_width = x, y
        self.obstacle_map = occ_map.copy()

    def Node(self, x, y, cost, parent_index):
        '''
        The basic node represening a location the robot can travel too
        :param x: X pos of robot
        :param y: Y pos of robot
        :param cost: Cost to travel to this target node
        :param parent_index: The node of the current robot position
        :return: A dictionary
        '''
        index = y * self.max_x + x  # Indexing a flattened matrix
        return {
            "x": x,
            "y": y,
            "cost": cost,
            "parent_index": parent_index,
            "id": index
        }

    def movement_pattern(self, num_pixels=1):
        '''
        The basic search pattern of the A* Algorithm. Basically searching every node around current node.
        :param num_pixels: The movement amount the A* algotihm takes around current node
        :return: A list of movements around current node, each with their associated costs
        '''
        dx, dy = num_pixels, num_pixels
        cost = math.sqrt(dx + dy)  # Simply euclidean distance
        return [[dx, 0, dx], [0, dy, dy], [-dx, 0, dx], [0, -dy, dy], [-dx, -dy, cost], [-dx, dy, cost], [dx, -dy, cost],[dx, dy, cost]]

    def close_enough(self, current, goal_node):
        '''
        Determines if robot is within radius of the goal
        :param current: The current position of the robot
        :param goal_node: The goal position the robot wishes to travel too
        :return: A boolean
        '''
        center_x, center_y = goal_node["x"], goal_node["y"]
        x, y = current["x"], current["y"]
        cond = (x - center_x)**2 + (y - center_y)**2 < self.goal_threshold**2
        return cond

    def get_path(self, sx, sy, gx, gy):
        '''
        Main diver of the A* algorithm. Determines a safe optimal path from current to goal position.
        :param sx: starting x position
        :param sy: starting y position
        :param gx: goal x position
        :param gy: goal y position
        :return: a list of points to the goal
        '''
        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)

        open_set = {}
        closed_set = {}
        open_set[start_node["id"]] = start_node

        while len(open_set) >= 1:

            def sortFunc(o):
                cost = open_set[o]['cost']
                heuristic = self.heuristic(goal_node, open_set[o])
                new_cost = cost + heuristic
                return new_cost

            c_id = min(open_set, key=lambda o: sortFunc(o))
            current = open_set[c_id]

            if self.close_enough(current, goal_node):
                goal_node['parent_index'] = current['parent_index']
                goal_node['cost'] = current['cost']
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            for i, (x, y, cost) in enumerate(self.motion):
                    node = self.Node(current['x'] + x, current['y'] + y, current['cost'] + cost, c_id)
                    n_id = node['id']

                    if (not self.verify_node(node, current)) or (n_id in closed_set):
                        continue

                    if n_id not in open_set:
                        open_set[n_id] = node  # discovered a new node
                    else:
                        if open_set[n_id]['cost'] > node['cost']:
                            # This path is the best until now. record it
                            open_set[n_id] = node

        path = []
        path.append([goal_node['x'], goal_node['y']])
        parent_index = goal_node['parent_index']
        while parent_index != -1:
            n = closed_set[parent_index]
            point = [n['x'], n['y']]
            path.append(point)
            parent_index = n['parent_index']

        return path


    def heuristic(self, n1, n2):
        '''
        This is the cost model of the A* algorithm.
        :param n1: Current node the robot is at
        :param n2: The target node the robot wants to travel too
        :return: A cost associated with traveling to the target node
        '''
        w = 1 if self.obstacle_map[n2['x']][n2['y']] == 0.0 else 2  # Prefer plotting path in known areas
        w = 5 if self.check_collision(n1, n2) else w  # Really punish running into a wall TODO: Try to find ways to remove this
        d = w * math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
        return d

    def too_close(self, point):

        '''
        Determine if there is an obstacle within the radius around point
        :param point: Point robot will move too
        :return: Boolean indicating if its safe to travel to this point
        '''
        x_lim, y_lim = self.obstacle_map.shape
        x, y = point
        x_start = int(np.clip(x - self.robot_radius, 0, x_lim))
        x_end = int(np.clip(x + self.robot_radius, 0, x_lim))
        y_start = int(np.clip(y - self.robot_radius, 0, y_lim))
        y_end = int(np.clip(y + self.robot_radius, 0, y_lim))
        roi = self.obstacle_map[x_start:x_end, y_start:y_end]

        if roi.max() < 1.0:
            return False
        else:
            return True

    def check_collision(self, n1, n2):

        '''
        :param n1: Starting point for the robot
        :param n2: Ending poinot for the roboot
        :return: boolean indicating if the robot will crash into something on the way to the goal
        '''

        ray_cast = bresenham((n1['x'], n1['y']), (n2['x'], n2['y']))
        # Overly redundant point checking
        # TODO: Makes point skip dynamic
        ray_cast = ray_cast[0: -1: 3] if len(ray_cast) > 10 else ray_cast
        for occupancy_point in ray_cast[0: -1: 2]:
            if self.obstacle_map[occupancy_point[0]][occupancy_point[1]] >= 1.0:
                return True
            elif self.too_close(occupancy_point):
                return True

        return False

    def verify_node(self, target, current):
        '''

        :param target: Target node
        :param current: Current node robot is located
        :return: A boolean indicating if target is safe to travel too
        '''
        if self.obstacle_map[target['x']][target['y']] >= 1.0:
            return False
        elif target['x'] < self.min_x:
            return False
        elif target['y'] < self.min_y:
            return False
        elif target['x'] >= self.max_x:
            return False
        elif target['y'] >= self.max_y:
            return False
        elif self.check_collision(current, target):
            return False
        else:
            return True
