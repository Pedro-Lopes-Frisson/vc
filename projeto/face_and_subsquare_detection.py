import math
import cv2
import numpy as np
import sys

low_threshold = 21
max_threshold = 45

def load_camera_calibration(filename, mtx,dist):
    with np.load("camera.npz") as data:
        mtx = data["intrinsics"]
        dist = data["distortion"]
    return mtx,dist

def find_cube_face(val):
    global low_threshold
    global max_threshold
    image = blurred
    low_threshold = val
    if image is None:
        return
    if len(image.shape) != 3:
        return None

    dst = cv2.Canny(image, low_threshold, max_threshold , None,3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)


    cv2.imshow("Canny Lines", cdst)
    cdstP = np.copy(cdst)
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 80, 8)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    return 0
def change_max_threshold(val):
    global max_threshold
    max_threshold = val


blurred = None
if __name__ == "__main__":
    cv2.namedWindow("Canny Lines", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Detected Lines (in red) - Probabilistic Line Transform", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("video", cv2.WINDOW_FREERATIO)
    cv2.createTrackbar("Min low_threshold:", "Canny Lines", low_threshold, 500, find_cube_face  )
    cv2.createTrackbar("Max threshold:", "Canny Lines", max_threshold, 500, change_max_threshold  )
    capture = cv2.VideoCapture("./20241123_182441.mp4")
    while (capture.isOpened()):
        ret, frame = capture.read()
        k = ''
        try:
            k = chr(cv2.waitKey(1)) 
        except:
            pass
        if k == 'q':
            break
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        find_cube_face(low_threshold)
        cv2.imshow('video', blurred)
