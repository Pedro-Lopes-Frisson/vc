import cv2
import numpy as np
import math
import sys

def apply_canny_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image,   (3,3), 0)
    dst = cv2.Canny(blurred, 20,30 , None)
    return dst

def dilate_countours(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    dilated = cv2.dilate(image, kernel, iterations=2)

    return dilated

def get_hough_lines(image):
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 50, None, 80, 8)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


def find_last_child_contours(contours, hierarchy):

    last_child_contours = []

    for idx, h in enumerate(hierarchy[0]):
        first_child = h[2]  # Index of the first child
        area = cv2.contourArea(contours[first_child])
        if area < 1000:
            last_child_contours.append(contours[idx])
            continue

        if first_child != -1:  # If the contour has a child
            last_child = first_child

            # Traverse the child chain to find the last child
            while hierarchy[0][last_child][0] != -1  :  # While there is a next sibling
                last_child = hierarchy[0][last_child][0]

            # Append the last child contour to the list
            last_child_contours.append(contours[last_child])
        else:
            last_child_contours.append(contours[idx])

    return last_child_contours
    
"""

green




"""

def reduce_contour_complexity(image,frame):
    countours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    image_ct = frame.copy()

    #last_child_contours = countours
    last_child_contours = find_last_child_contours(countours, hierarchy)

    for c in last_child_contours:

        area = cv2.contourArea(c)
        if area < 1000:
            continue
        

        cv2.drawContours(image_ct, c, -1, (0,255,0), thickness= 2)
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        #print(f"Before {c.shape=:} then {approx.shape=:}")

        
        if len(approx)  == 4:
            approx_contours.append(approx)
    
    return tuple(approx_contours)

if __name__ == "__main__":
    #cv2.namedWindow("Canny Lines", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Dilated Canny Lines", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("Detected Lines", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Image countours", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("HSV", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("image_ct", cv2.WINDOW_FREERATIO)

    #cv2.namedWindow("video", cv2.WINDOW_FREERATIO)

    green_lower=[54,63,79]
    green_upper=[68,222,157]
    yellow_lower=[22,106,178]
    yellow_upper=[39,222,245]
    red_lower=[3,101,25]
    red_upper=[5,255,207]
    blue_lower=[88,146,0]
    blue_upper=[179,239,186]
    orange_lower=[6,162,175]
    orange_upper=[17,255,255]
    white_lower=[16,0,216]
    white_upper=[70,39,255]



    capture = cv2.VideoCapture("./20241128_111931.mp4")
    #capture = cv2.VideoCapture("./20241123_182441.mp4")
    #capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture("http://192.168.241.75:4747/video")
    while (capture.isOpened()):
        ret, frame = capture.read()
        #frame =  cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
        k = ''
        try:
            k = chr(cv2.waitKey(1)) 
        except:
            pass
        if k == 'q':
            break

        if k == 'h':
            cv2.waitKey()
        print("\n\nNEW FRAME\n\n")
        original_frame = frame.copy()
        canny = apply_canny_detection(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        dilated_canny = dilate_countours(canny)

        #cv2.imshow( "Canny Lines",canny)
        cv2.imshow( "Dilated Canny Lines",dilated_canny)
        countours = reduce_contour_complexity(dilated_canny, frame)

        for approx in countours:
            #cv2.drawContours(frame, approx, -1, (255,0,0), thickness= 2)
            area = cv2.contourArea(approx)
            if area < 2000:
                continue
            rect = cv2.boundingRect(approx)
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + int(w * 0.9), y +int(h * 0.9)), (255, 0, 255), 2)  # Purple color
            color = np.zeros_like(frame[0,0])
            for point in approx:
                approx_x,approx_y = point[0]
                cv2.circle(frame, (approx_x,approx_y), 2, (255,255,255))

            cv2.putText(frame, f"{area}", (int(x+(w/2)), int(y+(h/2))), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.putText(frame, f"{len(approx)}", (int(x+(w/2)), int(y+(h/2) - 12)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.imshow( "Image countours",frame)
            #cv2.imshow("HSV", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

