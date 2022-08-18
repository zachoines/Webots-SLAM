import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cv2
import os

def get_edge_map(image):
    edge_image = cv2.GaussianBlur(image, (3, 3), 1)
    edge_image = np.round(edge_image)
    edge_image = np.uint8(edge_image * 255)
    edge_image = cv2.Canny(edge_image, 0, 255)
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    edge_image = cv2.erode(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    return edge_image


def detect_corners(image, print_corners=False):
    edges = get_edge_map(image)
    # features = cv2.goodFeaturesToTrack(
    #     edges,
    #     maxCorners=10,
    #     qualityLevel=.9,
    #     minDistance=10,
    #     useHarrisDetector=True
    #     )

    dst = cv2.cornerHarris(
        edges,
        blockSize=2,
        ksize=3,
        k=.04
    )
    dst = cv2.dilate(
        dst,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    dst = cv2.erode(
        dst,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    features = cv2.cornerSubPix(edges, np.float32(centroids), (5, 5), (-1, -1), criteria)
    results = []
    if len(features) > 0:
        # features = np.int0(features)

        tmp_image = image.copy() * 255
        for i in features:
            loc = i.ravel()
            results.append(loc)
            if print_corners:
                cv2.circle(tmp_image, np.int0(loc), 2, (255.0, 255.0, 0), -1)
                cv2.circle(edges, np.int0(loc), 4, (255.0, 255.0, 0), -1)

        if print_corners:
            figure = plt.figure(figsize=(12, 12))
            subplot1 = figure.add_subplot(1, 2, 1)
            subplot2 = figure.add_subplot(1, 2, 2)
            subplot1.imshow(tmp_image)
            subplot2.imshow(edges, cmap="gray")
            plt.show()

    return results


def hough_line_detection_lidar(xs, ys, radius=1, num_rhos=180, num_thetas=180, t_count=100):
    thetas = np.arange(0, np.pi, np.pi / num_thetas)
    rs = np.arange(-radius, radius, radius / num_rhos)

    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    accumulator = np.zeros((len(rs), len(thetas)))

    for x, y, in zip(xs, ys):
        plot_ys, plot_xs = [], []
        for t_j, phi in enumerate(thetas):

            # Determine the parameters of perpendicular line
            r = (x * cos_thetas[t_j]) + (y * sin_thetas[t_j])

            # Find the closest fitting line in the Hough Space
            r_i = np.argmin(np.abs(rs - r))

            # Mark as found
            accumulator[r_i][t_j] += 1
            plot_ys.append(r)
            plot_xs.append(phi)

    detected_lines = []
    for y, rad in enumerate(rs):
        for x, theta in enumerate(thetas):
          if accumulator[y][x] - t_count > 0:
            detected_lines.append((rad, theta))

    return detected_lines

def hough_line_detection_image(image, edge_image, num_rhos=400, num_thetas=700, t_count=100, plot_lines=True):
    # https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549

    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos

    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    edge_points = np.argwhere(edge_image != 0)
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))

    accumulator, theta_vals, rho_vals = np.histogram2d(
      np.tile(thetas, rho_values.shape[0]),
      rho_values.ravel(),
      bins=[thetas, rhos]
    )
    accumulator = np.transpose(accumulator)
    lines = np.argwhere(accumulator > t_count)
    rho_idxs, theta_idxs = lines[:, 0], lines[:, 1]
    r, t = rhos[rho_idxs], thetas[theta_idxs]

    if plot_lines:

        figure = plt.figure(figsize=(12, 12))
        subplot1 = figure.add_subplot(1, 4, 1)
        subplot1.imshow(image)
        subplot2 = figure.add_subplot(1, 4, 2)
        subplot2.imshow(edge_image, cmap="gray")
        subplot3 = figure.add_subplot(1, 4, 3)
        subplot3.set_facecolor((0, 0, 0))
        subplot4 = figure.add_subplot(1, 4, 4)
        subplot4.imshow(image)

        for ys in rho_values:
            subplot3.plot(thetas, ys, color="white", alpha=0.05)

        subplot3.plot([t], [r], color="yellow", marker='o')

        for rho, theta in zip(r, t):
            a = np.cos(np.deg2rad(theta))
            b = np.sin(np.deg2rad(theta))
            x0 = (a * rho) + edge_width_half
            y0 = (b * rho) + edge_height_half
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            subplot3.plot([theta], [rho], marker='o', color="yellow")
            subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

        subplot3.invert_yaxis()
        subplot3.invert_xaxis()

        subplot1.title.set_text("Original Image")
        subplot2.title.set_text("Edge Image")
        subplot3.title.set_text("Hough Space")
        subplot4.title.set_text("Detected Lines")
        plt.show()

    return [[r_i, t_i] for r_i, t_i in zip(r, t)]

def calibrate_camera(path="./calibration_images/"):
    # Define the dimensions of checkerboard.
    CHECKERBOARD = (9, 9)  # W x H

    # Termination criteria.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_corners = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    obj_corners[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []
    img_points = []

    # Store undistortion results.
    camera_matrix = None
    distortion_coefficients = None
    rotation_vectors = None
    translation_vectors = None
    new_camera_matrix = None
    roi = None

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        image = cv2.imread(f)

        # Receive the frame.
        image = image.astype(np.float32)
        # image = cv2.transpose(image)
        # cv2.imshow('Bitwise Checkerboard Image', image / 255.0)
        # camera.saveImage("checkerboard_1.png", 0)


        # Color-segmentation to get binary mask
        # lwr = np.array([0, 0, 143])
        # upr = np.array([179, 61, 252])
        lwr = np.array([0,0,128], dtype=np.uint8)
        upr = np.array([255,255,255], dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, lwr, upr)

        # Extract chess-board
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dlt = cv2.dilate(msk, krn, iterations=5)


        res = cv2.bitwise_and(dlt, msk)
        # cv2.imshow('Detected Checkerboard Pattern', res)
        checkerboard = np.uint8(res)

        # Find and draw the checkboard corners.
        is_pattern_found, img_corners = cv2.findChessboardCorners(checkerboard, CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                           cv2.CALIB_CB_FAST_CHECK +
                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
        image = cv2.drawChessboardCorners(image / 255.0, CHECKERBOARD, img_corners, is_pattern_found)
        cv2.imshow('Detected Checkerboard Pattern', image)

        # If found, add object points and image points.
        if is_pattern_found:
            obj_points.append(obj_corners)
            img_points.append(img_corners)

            # Check if we now have enough points to undistort.
            if len(img_points[0]) >= 3:
                _, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
                    obj_points, img_points, (1080, 1080), None, None)
                # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients,
                #                                                        image.shape[::-1], 1,
                #                                                        image.shape[::-1])

        # Check if we can undistort this frame.
        # if camera_matrix is not None:
        #     image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
        #     x, y, w, h = roi
        #     image = image[y:y + h, x:x + w]

        # Display the video feed.
        # cv2.imshow('Video', image)
        #
        # # Check if a key was pressed.
        # key = cv2.waitKey(1)
        # if key == 32:
        #     # If spacebar was pressed, capture the next image.
        #     is_capturing = True
        # elif key == 8:
        #     # If Backspace was pressed, delete the last image.
        #     del obj_points[-1]
        #     del img_points[-1]
        #     cv2.destroyWindow('Captured Image')
        # elif key == 27:
        #     # If Escape was pressed, stop capturing images.
        #     cv2.destroyAllWindows()
        #     break

    return camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, new_camera_matrix, roi