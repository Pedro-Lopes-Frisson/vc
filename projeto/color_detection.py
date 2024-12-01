import cv2
import numpy as np
import math
import sys


def nothing(args):
    pass


if __name__ == "__main__":
    cv2.namedWindow("Video", cv2.WINDOW_FREERATIO)


    cv2.createTrackbar('HMin', 'Video', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'Video', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'Video', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'Video', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'Video', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'Video', 0, 255, nothing)

    capture = cv2.VideoCapture("./20241128_111931.mp4")
    #capture = cv2.VideoCapture("./20241123_182441.mp4")
    #capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture("http://192.168.241.75:4747/video")

    ret, frame = capture.read()
    while (capture.isOpened()):
        hMin = cv2.getTrackbarPos('HMin', 'Video')
        sMin = cv2.getTrackbarPos('SMin', 'Video')
        vMin = cv2.getTrackbarPos('VMin', 'Video')
        hMax = cv2.getTrackbarPos('HMax', 'Video')
        sMax = cv2.getTrackbarPos('SMax', 'Video')
        vMax = cv2.getTrackbarPos('VMax', 'Video')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        k = ''
        try:
            k = chr(cv2.waitKey(1)) 
        except:
            pass
        if k == 'q':
            break
        if k == 'r':
            ret, frame = capture.read()
        if k == 'p':
            print(f"{lower=:}")
            print(f"{upper=:}")


        if k == 'h':
            cv2.waitKey()

        cv2.imshow('Video', result)

