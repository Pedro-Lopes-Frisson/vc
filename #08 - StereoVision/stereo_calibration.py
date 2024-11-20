# Cheesboard.py
#
# Chessboard Calibration
#
# Paulo Dias
#
# Pedro Lopes

import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


def FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    # If found, display image with corners
    if ret == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)

    return ret, corners


def FindImagePoints(imgpoints, filesStr, imageSize, objpoints = []):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.

    # Read images
    images = sorted(glob.glob(filesStr))
    print(images)

    for fname in images:
        img = cv2.imread(fname)
        ret, corners = FindAndDisplayChessboard(img)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return img.shape[:2]



left_corners = []
right_corners = []
objpoints = []
imageSize = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
stereocalibration_flags = cv2.CALIB_SAME_FOCAL_LENGTH 


FindImagePoints(left_corners, "..//images//left*.jpg", imageSize)

"""
ret, mtx1, dist1, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, left_corners, imageSize, None, None
)
"""


imageSize = FindImagePoints(right_corners, "..//images//right*.jpg", imageSize, objpoints)
"""
ret, mtx2, dist2, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, right_corners, imageSize, None, None
)
"""


print(f"{imageSize=:}")
retval ,mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    left_corners,
    right_corners,
    cameraMatrix1=None,
    cameraMatrix2=None,
    distCoeffs1=None,
    distCoeffs2=None,
    imageSize=imageSize,
    criteria=criteria,
    flags=stereocalibration_flags,
)

np.savez(
    "stereoParams.npz",
    intrinsics1=mtx1,
    distortion1=dist1,
    intrinsics2=mtx2,
    distortion2=dist2,
    R=R,
    T=T,
    E=E,
    F=F,
)

print(f"{retval=:}")
print(f"{mtx1=:}")
print(f"{dist1=:}")
print(f"{mtx2=:}")
print(f"{dist2=:}")
print(f"{R=:}")
print(f"{T=:}")
print(f"{E=:}")
print(f"{F=:}")
