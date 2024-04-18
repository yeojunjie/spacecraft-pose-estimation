import pandas as pd
import numpy as np
from utils import * # ⚠️⚠️ The functions in utils.py are not correct.
import cv2 as cv
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Whenever the range data is not available, we first use the most recently known value.
# If that is not available (e.g. the first image in the chain has missing range data), we use the placeholder value.
# The problem constrains the camera to be at least 20 metres and at most 600 metres away from the spacecraft.
ESTIMATED_DISTANCE_TO_SPACECRAFT = (20 + 600) / 2 # metres

# The intrinsic camera matrix.
# Source: https://www.drivendata.org/competitions/261/spacecraft-pose-estimation/page/834/#camera-intrinsic-parameters
K = np.array([[5.2125371e+03, 0.0000000e+00, 6.4000000e+02],
              [0.0000000e+00, 6.2550444e+03, 5.1200000e+02],
              [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])

def process_one_chain_using_sift_ransac(absolute_path_to_chain_folder, absolute_path_to_range_file, file_to_append_to):
    CHAIN_ID = absolute_path_to_chain_folder.split('/')[-1]

    # Load the images.
    images = []
    for i in range(0, 100):
        image_path = os.path.join(absolute_path_to_chain_folder, f'{i:03}.png')
        if os.path.exists(image_path):
            images.append(cv.imread(image_path, cv.IMREAD_GRAYSCALE))
        else:
            raise Exception(f'The image {image_path} does not exist.')
    
    # Load the range data.
    # NOTE: This range data file is massive. It contains information for every single chain in the dataset.
    # Possible optimisation: Use interpolation search for the correct row index, as range.csv is sorted.
    range_data = pd.read_csv(absolute_path_to_range_file, header=None)
    ranges = range_data.loc[range_data[0] == CHAIN_ID][2].astype("float").values

    last_known_range = ESTIMATED_DISTANCE_TO_SPACECRAFT

    # The reference image requires no transformation.
    with open(file_to_append_to, 'a') as file:
        # Header: chain_id,i,range,x,y,z,qw,qx,qy,qz
        file.write(f'{CHAIN_ID},0,{ranges[0]},0,0,0,1,0,0,0\n')

    # For every pair of images (i, i+1),...
    # See main.ipynb for more detailed comments.
    for i in range(0, 99):

        # Update the range data if it is available.
        if np.isnan(ranges[i+1]) == False:
            last_known_range = ranges[i+1]

        # Detect features.
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(images[i+1], None)
        kp2, des2 = sift.detectAndCompute(images[i], None)

        # Error handling.
        if des1 is None or des2 is None:
            # If there are no features, we assume that the image did not move.
            with open(file_to_append_to, 'a') as file:
                # Header: chain_id,i,range,x,y,z,qw,qx,qy,qz
                file.write(f'{CHAIN_ID},{i+1},{last_known_range},0,0,0,1,0,0,0\n')
            continue

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
        with open(file_to_append_to, 'a') as file:
            # Header: chain_id,i,range,x,y,z,qw,qx,qy,qz
            file.write(f'{CHAIN_ID},{i+1},{last_known_range},{T[0]},{T[1]},{T[2]},{rotation_wxyz[0]},{rotation_wxyz[1]},{rotation_wxyz[2]},{rotation_wxyz[3]}\n')



        