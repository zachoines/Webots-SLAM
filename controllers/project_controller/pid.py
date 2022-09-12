import numpy as np


class PID:
    def __init__(self, target, max_velocity, max_acceleration, p, i, d, windup_gaurd=1.0):
        self.target = target
        self.P = p
        self.I = i
        self.D = d
        self.integral = 0
        self.previous_error = 0
        self.previous_velocity = 0
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.windup_gaurd = windup_gaurd

    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_velocity = 0

    def updateTarget(self, target):
        self.target = target

    def update(self, current, ts, reverse=False):
        error = current - self.target if reverse else self.target - current
        self.integral += error * ts

        # Clip integral
        if abs(self.integral) > self.windup_gaurd:
            self.integral = np.sign(self.integral) * self.windup_gaurd

        error_derivative = (self.previous_error - error) / ts

        # Determine new control velocity
        new_velocity = (self.P * error) + (self.D * error_derivative) + (self.I * self.integral)

        # Clip velocity
        if abs(new_velocity) > self.max_velocity:
            new_velocity = np.sign(new_velocity) * self.max_velocity

        # Clip acceleration
        a = (new_velocity - self.previous_velocity) / ts

        if abs(a) > self.max_acceleration:
            a = np.sign(a) * self.max_acceleration
        new_velocity = self.previous_velocity + a * ts

        self.previous_velocity = new_velocity
        self.previous_error = error
        # print("P: " + str(self.P * error) + ", I: " + str(self.integral) + ", D: " + str(error_derivative))
        return new_velocity

