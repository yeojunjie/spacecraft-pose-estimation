import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# The intrinsic camera matrix.
# Source: https://www.drivendata.org/competitions/261/spacecraft-pose-estimation/page/834/#camera-intrinsic-parameters
K = np.array([[5.2125371e+03, 0.0000000e+00, 6.4000000e+02],
              [0.0000000e+00, 6.2550444e+03, 5.1200000e+02],
              [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])

IDENTITY_TRANSLATION = np.array([0, 0, 0])
IDENTITY_ROTATION = np.array([1, 0, 0, 0])

def predict_transformation_between_two_images(from_image, to_image):
    """
    Predict the transformation between two images using SIFT and RANSAC.
    """

    # Detect features.
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(from_image, None)
    kp2, des2 = sift.detectAndCompute(to_image, None)

    # Error handling.
    if des1 is None or des2 is None:
        # If there are no features, we assume that the image did not move.
        return IDENTITY_ROTATION, IDENTITY_TRANSLATION

    # Match features.
    bf = cv.BFMatcher()
    pairs_of_matches = bf.knnMatch(des1, des2, k=2)

    # Select unambiguous matches.
    good_matches = []
    for pair_of_matches in pairs_of_matches:

        # Sometimes, knnMatch can only find one match, so there is no 'next best match'.
        # For this reason, the match is unambiguous, and is therefore good.
        if len(pair_of_matches) == 1:
            good_matches.append(pair_of_matches[0])
            continue

        m, n = pair_of_matches
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract the matched keypoints.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Error handling.
    if len(src_pts) == 0:
        src_pts = np.zeros((1, 1, 2))
        dst_pts = np.zeros((1, 1, 2))

    # Extract the fundamental matrix.
    F, mask = cv.findFundamentalMat(src_pts, dst_pts, cv.FM_RANSAC)

    # Error handling.
    if F is None:           # A solution could not be found.
        F = np.eye(3)
    if F.shape == (9, 3):   # Three solutions were found.
        F = F[:3]
    
    # Adjust for the camera's distortion.
    E = K.T @ F @ K

    # Extract the rotation and translation.
    _, R, T, _ = cv.recoverPose(E, src_pts, dst_pts, K)
    T = T.flatten()

    # Convert the rotation matrix to a quaternion.
    rotation_xyzw = Rotation.from_matrix(R).as_quat()
    rotation_wxyz = np.roll(rotation_xyzw, 1)

    # Write the results to the file.
    return rotation_wxyz, T