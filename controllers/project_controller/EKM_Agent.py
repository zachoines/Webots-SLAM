import numpy as np
import cv2
from config import *
from utility import flatten_list
from vector import distance

class EKF_Agent:
    def __init__(self, initial_x, max_landmarks):
        self.max_landmarks = max_landmarks
        self.landmark_state_size = self.max_landmarks * 3  # 3 -- [x, y, z] of landmark
        self.full_state_size = ROBOT_STATE['SIZE'] + self.landmark_state_size

        # Jacobians and state matrix
        self.x_hat_t = initial_x + [0.0 for _ in range(self.landmark_state_size)]
        self.phi = np.eye(self.full_state_size)
        self.g = np.zeros((self.full_state_size, 2))

        # Covariance matrices
        self.sigma_x_t = np.eye(self.full_state_size)
        all_cov = [STD_X[i] * STD_X[i] for i in range(len(STD_X))] + [STD_L**2 for _ in range(self.landmark_state_size)]
        self.sigma_x_t[np.diag_indices(self.full_state_size)] = all_cov
        self.sigma_m = np.eye(3 * NUM_LANDMARKS) * (STD_M[0]**2)
        self.sigma_n = np.array([[STD_N[0]**2, 0], [0, STD_N[1]**2]])
        self.landmarks_track_count = np.zeros((NUM_LANDMARKS))


    def _f(self, U, dt):
        '''
            State prop func
        '''

        theta = self.x_hat_t[ROBOT_STATE["THETA"]]
        v = U[CONTROL_SIGNALS["V"]]
        w = U[CONTROL_SIGNALS["W"]]

        self.x_hat_t[ROBOT_STATE["X"]] += dt * v * np.cos(theta)
        self.x_hat_t[ROBOT_STATE["Y"]] += dt * v * np.sin(theta)
        self.x_hat_t[ROBOT_STATE["Z"]] = self.x_hat_t[ROBOT_STATE["Z"]]  # z never changes
        self.x_hat_t[ROBOT_STATE["THETA"]] += dt * w

        return self.x_hat_t

    def _PHI(self, U, dt):
        '''
            Jacobian of the f() with respect to X
            Pg. 108-109 of Stergios notes.
            https://www.wolframalpha.com/input?i=jacobian+of+%28x+%2B+%28t+*+v+*+cos%28%CF%95%29%29%2C+y+%2B+%28t+*+v+*+sin%28%CF%95%29%29%2C+z%2C+%CF%95+%2B+%28t+*+w%29%29+with+respect+to+%28x%2C+y%2C+z%2C+%CF%95%29%2C+%29
        '''

        theta = self.x_hat_t[ROBOT_STATE["THETA"]]
        v = U[CONTROL_SIGNALS["V"]]

        self.phi[:ROBOT_STATE["SIZE"], :ROBOT_STATE["SIZE"]] = np.array([
            [1, 0, 0, -dt * v * np.sin(theta)],
            [0, 1, 0, dt * v * np.cos(theta)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return self.phi

    def _G(self, dt):
        '''
            Jacobian of f() with respect to the noise 'w'.
            Pg. 108-109 of Stergios notes.
        '''

        theta = self.x_hat_t[ROBOT_STATE["THETA"]]

        self.g[:ROBOT_STATE["SIZE"], :2] = np.array([
            [-dt * np.cos(theta), 0],
            [-dt * np.sin(theta), 0],
            [0, 0],
            [0, -dt]
        ])

        return self.g

    def _equal(self, l1, l2):
        return np.alltrue(np.equal(np.array(l1), np.array(l2)))

    def _list_null(self, l):
        return self._equal(l, [0., 0., 0.])

    def _update_covs(self, landmark_loc):
        # TODO: If we detect a new landmark and wish to replace an old one, we need to update our covariance matrices
        # index = ROBOT_STATE["THETA"] + 1 + landmark_loc
        # self.sigma_x_t[index, index] = STD_L
        # self.sigma_x_t[0, index] = 0
        pass

    def _align_landmarks_and_measurements(self, previous_landmarks, new_landmarks, new_measurements, y=0.01, stale_thresh=5):
        updated_landmarks = []
        updated_measurements = []
        update_mask = np.ones((NUM_LANDMARKS, LANDMARK_STATE_SIZE))
        previous_landmarks = np.array(previous_landmarks).tolist()
        new_landmarks = np.array(new_landmarks).tolist()
        new_measurements = np.array(new_measurements).tolist()
        for i, l1 in enumerate(previous_landmarks):
            count = len(new_landmarks)
            null = self._list_null(l1)
            best = (np.Infinity, None, None, False)

            # Case where we can simply add the new landmark
            if null and count > 0:
                self.landmarks_track_count[i] = 0
                updated_landmarks.append(new_landmarks[0])
                updated_measurements.append(new_measurements[0])
                new_landmarks.remove(new_landmarks[0])
                new_measurements.remove(new_measurements[0])
                continue

            # There are no detected landmarks
            if count <= 0:
                self.landmarks_track_count[i] += 1
                updated_landmarks.append(l1)
                updated_measurements.append([0, 0, 0])
                update_mask[i, :] = np.array([0, 0, 0])
                continue

            # Find the closest landmark
            elif not null:
                for l2, m in zip(new_landmarks, new_measurements):
                    dist = distance(np.array(l1), np.array(l2))

                    # We found the landmark again
                    if dist < best[0]:
                        best = (dist, l2, m, True)

            dist, landmark, measurement, found = best

            # Case where we can replace old landmark
            if found and dist >= y and self.landmarks_track_count[i] >= stale_thresh:
                self.landmarks_track_count[i] = 0
                self._update_covs(i)
                updated_landmarks.append(landmark)
                updated_measurements.append(measurement)
                new_landmarks.remove(landmark)
                new_measurements.remove(measurement)

            # Case where landmark is out of sight, so we place an update mask
            elif found and dist >= y and self.landmarks_track_count[i] < stale_thresh:
                self.landmarks_track_count[i] += 1
                updated_landmarks.append(l1)
                updated_measurements.append([0, 0, 0])
                update_mask[i, :] = np.array([0, 0, 0])

            # Case where we found previous landmark
            elif found and dist < y:
                self.landmarks_track_count[i] = 0
                updated_landmarks.append(landmark)
                updated_measurements.append(measurement)
                new_landmarks.remove(landmark)
                new_measurements.remove(measurement)

            # Probably shouldn't come to this....
            else:
                self.landmarks_track_count[i] += 1
                updated_landmarks.append(l1)
                updated_measurements.append([0, 0, 0])
                update_mask[i, :] = np.array([0, 0, 0])

        assert len(updated_landmarks) == NUM_LANDMARKS

        return np.array(updated_landmarks), np.array(updated_measurements), update_mask

    def _update_landmarks(self, z_t, g_p_l):
        # NOTE: Simply use euclidean distance to track landmarks for now
        # TODO:: Need to devise a better way to track landmarks across frames here
        g_p_l, z_t, update_mask = self._align_landmarks_and_measurements(
            np.array_split(np.array(self.x_hat_t[ROBOT_STATE["THETA"] + 1:]), NUM_LANDMARKS),
            g_p_l,
            z_t
        )

        self.x_hat_t[ROBOT_STATE["THETA"]+1:] = np.array(g_p_l).flatten()
        return z_t, g_p_l, update_mask

    def _h(self, all_g_pos_l):
        '''
        Measurement function. Returns Relative position of the landmark
        with respect to the robot's local frame.
        Pg. 115 of Stergios notes.
        '''
        theta = self.x_hat_t[ROBOT_STATE["THETA"]]
        g_p_r = self.x_hat_t[ROBOT_STATE["X"]:ROBOT_STATE["Z"] + 1]
        g_rot_r, _ = cv2.Rodrigues(np.array([0, 0, theta]))
        r_pos_l = [g_rot_r.T @ (g_pos_l - g_p_r) for g_pos_l in all_g_pos_l]

        return np.array(r_pos_l)

    def _H(self, all_w_pos_l):


        '''
            Jacobian of the measurement
            Pg. 117 of Stergios notes.

            [[x2 - x1],[ y2 - y1], [z2 - z1], []]

            Help-full References:
            https://www.wolframalpha.com/input?i=dot+product&assumption=%7B%22C%22%2C+%22dot+product%22%7D+-%3E+%7B%22Calculator%22%7D&assumption=%7B%22F%22%2C+%22DotProduct%22%2C+%22dotVector1%22%7D+-%3E%22transpose%28%5B%5Bcos%28phi%29%2C+-sin%28phi%29%2C+0%5D%2C+%5Bsin%28phi%29%2C+cos%28phi%29%2C+0%5D%2C+%5B0%2C+0%2C+1%5D%5D%29%22&assumption=%7B%22F%22%2C+%22DotProduct%22%2C+%22dotVector2%22%7D+-%3E%22%5B%5Bx2+-+x1%5D%2C%5B+y2+-+y1%5D%2C+%5Bz2+-+z1%5D%5D%22
             H(z) = r_pos_l = Rot(z).T @ (w_pos_l - w_pos_r) =

            [
                (-x1 + x2) cos(ϕ) + (-y1 + y2) sin(ϕ),
                (-y1 + y2) cos(ϕ) - (-x1 + x2) sin(ϕ),
                 -z1 + z2
            ]

            https://www.wolframalpha.com/input?i=jacobian+of+%28%28-x1+%2B+x2%29+cos%28%CF%95%29+%2B+%28-y1+%2B+y2%29+sin%28%CF%95%29%2C+%28-y1+%2B+y2%29+cos%28%CF%95%29+-+%28-x1+%2B+x2%29+sin%28%CF%95%29%2C+-z1+%2B+z2%29+with+respect+to+%28x1%2C+y1%2C+z1%2C+%CF%95%2C+x2%2C+y2%2C+z2%29%29
        The Jacobian w.r.t to x:
        '''
        theta = self.x_hat_t[ROBOT_STATE["THETA"]]
        g_p_r = self.x_hat_t[ROBOT_STATE["X"]:ROBOT_STATE["Z"] + 1]
        diffs = np.array(all_w_pos_l) - np.array(g_p_r)

        H = np.array(
            [
                [
                    [-np.cos(theta), -np.sin(theta), 0, ((diff[1] * np.cos(theta)) - (diff[0] * np.sin(theta)))] + flatten_list([[np.cos(theta), np.sin(theta), 0] for _ in range(NUM_LANDMARKS)]),
                    [np.sin(theta), -np.cos(theta), 0, ((-diff[0] * np.cos(theta)) - (diff[1] * np.sin(theta)))] + flatten_list([[np.sin(theta), np.cos(theta), 0] for _ in range(NUM_LANDMARKS)]),
                    [0, 0, -1, 0] + flatten_list([[0, 0, 1] for _ in range(NUM_LANDMARKS)])
                ] for diff in diffs
            ]
        )

        return H.reshape((self.landmark_state_size, H.shape[-1]))


    def update(self, z_t, g_p_l):
        '''
            x_hat_t -- robot position, orientation, and landmarks
            sigma_x_t -- state estimation uncertainty
            z_t -- measurements
            z_hat_t -- estimated measurement
            sigma_m -- measurements' uncertainty
            g_p_l -- landmarks' global positions
            r_t -- error between prediction and measurement
            K_t -- Kalman filter
            S_t --
        '''

        z_t, g_p_l, update_mask = self._update_landmarks(z_t, g_p_l)

        z_hat_t = self._h(g_p_l)

        r_t = np.squeeze((z_t - z_hat_t) * update_mask).flatten()

        H_t = self._H(g_p_l)

        S_t = H_t @ self.sigma_x_t @ H_t.T + self.sigma_m

        K_t = self.sigma_x_t @ H_t.T @ np.linalg.inv(S_t)

        self.x_hat_t = self.x_hat_t + K_t @ r_t

        self.sigma_x_t = self.sigma_x_t - K_t @ H_t @ self.sigma_x_t  # Pg. 97
        # self.sigma_x_t = self.sigma_x_t - K_t @ S_t @ K_t.T  # Pg. 130

        return self.x_hat_t, self.sigma_x_t

    def propagate(self, u, dt):
        '''

        x_hat_t -- robot position and orientation
        sigma_x_t -- estimation uncertainty
        u -- control signals
        sigma_n -- uncertainty in control signals
        dt -- timestep

        '''

        self._PHI(u, dt)

        self._G(dt)

        self._f(u, dt)

        self.sigma_x_t = self.phi @ self.sigma_x_t @ self.phi.T + self.g @ self.sigma_n @ self.g.T

        return self.x_hat_t, self.sigma_x_t