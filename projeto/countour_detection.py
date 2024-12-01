import math
import cv2
import numpy as np
import sys

low_threshold = 31
max_threshold = 101
frame = None


def load_camera_calibration(filename, mtx,dist):
    with np.load("camera.npz") as data:
        mtx = data["intrinsics"]
        dist = data["distortion"]
    return mtx,dist

def find_cube_face(val):
    global low_threshold
    global max_threshold
    blue, green, red = cv2.split(frame)
# detect contours using blue channel and without thresholding
    contours1, hierarchy1 = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
     
# draw contours on the original image
    image_contour_blue = frame.copy()
    cv2.drawContours(image=image_contour_blue, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# see the results
    cv2.imshow('Contour detection using blue channels only', image_contour_blue)
     
# detect contours using green channel and without thresholding
    contours2, hierarchy2 = cv2.findContours(image=green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
    image_contour_green = frame.copy()
    cv2.drawContours(image=image_contour_green, contours=contours2, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# see the results
    cv2.imshow('Contour detection using green channels only', image_contour_green)
     
# detect contours using red channel and without thresholding
    contours3, hierarchy3 = cv2.findContours(image=red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
    image_contour_red = frame.copy()
    cv2.drawContours(image=image_contour_red, contours=contours3, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# see the results
    cv2.imshow('Contour detection using red channels only', image_contour_red)



    frame_og = frame.copy()
    frame_c = cv2.cvtColor(frame_og, cv2.COLOR_BGR2GRAY)
# detect contours using red channel and without thresholding
    contours4, hierarchy4 = cv2.findContours(image=frame_c, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
    cv2.drawContours(image=frame_og, contours=contours4, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# see the results
    cv2.imshow('Contour detection using 3 channels ', frame_og)
    
    return 0


def change_max_threshold(val):
    global max_threshold
    max_threshold = val


if __name__ == "__main__":
    cv2.namedWindow("Contour detection using red channels only", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Contour detection using green channels only", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Contour detection using blue channels only", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Contour detection using 3 channels ", cv2.WINDOW_FREERATIO)

    cv2.namedWindow("video", cv2.WINDOW_FREERATIO)

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

        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        find_cube_face(low_threshold)
        cv2.imshow('video', frame)

