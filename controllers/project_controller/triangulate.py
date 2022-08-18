import numpy as np
import math

def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


def project_on_image(position, fov_x=0.84, dims=(1080, 1080)):
    x, y, z = position
    w, h = dims
    aspect_ratio = h / w
    fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / aspect_ratio)
    theta1 = -math.atan2(y, abs(x))
    theta2 = math.atan2(z, abs(x))
    u = w * (0.5 * math.tan(theta1) / math.tan(0.5 * fov_x) + 0.5)
    v = h * (-0.5 * math.tan(theta2) / math.tan(0.5 * fov_y) + 0.5)
    u = max(0, min(u, w - 1))
    v = max(0, min(v, h - 1))
    return u, v

def least_squares(a, b):
    return np.linalg.lstsq(a, b, rcond=-1)[0]


def get_xyz(camera1_coords, camera1_M, camera1_R, camera1_T, camera2_coords, camera2_M, camera2_R, camera2_T):
    # Get the two key equations from camera1
    camera1_u, camera1_v = camera1_coords
    # Put the rotation and translation side by side and then multiply with camera matrix
    tmp = np.column_stack((camera1_R, camera1_T))
    camera1_P = camera1_M.dot(tmp)
    # Get the two linearly independent equation referenced in the notes
    camera1_vect1 = camera1_v * camera1_P[2, :] - camera1_P[1, :]
    camera1_vect2 = camera1_P[0, :] - camera1_u * camera1_P[2, :]

    # Get the two key equations from camera2
    camera2_u, camera2_v = camera2_coords
    # Put the rotation and translation side by side and then multiply with camera matrix
    camera2_P = camera2_M.dot(np.column_stack((camera2_R, camera2_T)))
    # Get the two linearly independent equation referenced in the notes
    camera2_vect1 = camera2_v * camera2_P[2, :] - camera2_P[1, :]
    camera2_vect2 = camera2_P[0, :] - camera2_u * camera2_P[2, :]

    # Stack the 4 rows to create one 4x3 matrix
    full_matrix = np.row_stack((camera1_vect1, camera1_vect2, camera2_vect1, camera2_vect2))
    # The first three columns make up A and the last column is b
    A = full_matrix[:, :3]
    b = full_matrix[:, 3].reshape((4, 1))
    # Solve overdetermined system. Note b in the wikipedia article is -b here.
    # https://en.wikipedia.org/wiki/Overdetermined_system
    soln = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(-b)
    return soln


def make_skew_symmetric_matrix(vec3d):
    a, b, c = vec3d
    skew_symmetric = np.asarray([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0],
    ])
    return skew_symmetric


def triangulation(P1, P2, pts1, pts2):
    pts3D = []
    for pt1, pt2 in zip(pts1, pts2):
        pt1_3d = list(pt1) + [1]
        pt2_3d = list(pt2) + [1]
        pt1_skew_symmteric = make_skew_symmetric_matrix(pt1_3d)
        pt2_skew_symmteric = make_skew_symmetric_matrix(pt2_3d)
        pt1_cross_P1 = pt1_skew_symmteric @ P1
        pt2_cross_P2 = pt2_skew_symmteric @ P2
        A = np.vstack((pt1_cross_P1[:2], pt2_cross_P2[:2]))
        X = null_space(A, rcond=1)
        # Take the first null space entry
        X = X[:, 0]
        # Divide by w
        X = X / X[-1]
        pts3D.append(X[:3])
    pts3D = np.asarray(pts3D)
    return pts3D



def triangulate_landmark(local_camera_poses, points, extrinsic_matrices, P, K, all=False):
    # http://16385.courses.cs.cmu.edu/spring2022/lecture/stereogeometry/

    K_inv = np.linalg.inv(K)
    # K_inv = np.column_stack((K_inv, [0, 0, 0]))
    K = np.column_stack((K, [0, 0, 0]))
    point_world = np.array([0.06, 1.19, 0.12, 1])

    for p, e, c in zip(points, extrinsic_matrices, local_camera_poses):
        image_cords = project_on_image(c[0:3], fov_x=0.84, dims=(1080, 1080))
        # homogenous_pixel_coords = invert_homogenous_transformation(e) @ K @ c
        # pixels = homogenous_pixel_coords[:-1] / homogenous_pixel_coords[-1]
        hello = "world"
    if not all:
        P1 = K @ computePose(extrinsic_matrices[0])
        P2 = K @ computePose(extrinsic_matrices[-1])
        point1 = points[0]
        point2 = points[-1]
        # triangulated = triangulation(P1, P2, [point1], [point2])
        A = [
            point1[1] * P1[2, :] - P1[1, :],
            P1[0, :] - point1[0] * P1[2, :],
            point2[1] * P2[2, :] - P2[1, :],
            P2[0, :] - point2[0] * P2[2, :]
        ]
        A = np.array(A).reshape((4, 4))
        B = A.transpose() @ A
        u, s, vh = np.linalg.svd(B)
        vect = vh[:, -1]
        triangulated = vect[0:-1] / vect[-1]
        triangulated2 = vh[:, 0][0:-1] / vh[:, 0][-1]
        return triangulated
    else:
        ps = [K @ e for e in extrinsic_matrices]
        A = []
        for p, (u, v) in zip(ps, points):
            A.append(v * p[2, :] - p[1, :])
            A.append(p[0, :] - u * p[2, :])
        A = np.array(A)
        eigen_values, eigen_vectors = np.linalg.eig(A.T @ A)
        idx = eigen_values.argsort()[::-1]
        eigen_vectors = eigen_vectors[:, idx]
        smallest_eigen_vector = eigen_vectors[-1]
        triangulated = smallest_eigen_vector[0:-1] / smallest_eigen_vector[-1]
        return triangulated

