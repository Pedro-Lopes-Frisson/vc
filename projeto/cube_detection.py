import cv2
import numpy as np
import math
import sys


def apply_canny_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image,   (3,7), 0)
    dst = cv2.Canny(blurred, 20,50 , None)
    return dst

def dilate_contours(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    dilated = cv2.dilate(image, kernel, iterations=4)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return dilated

def detect_closest_nine_squares(contours, gray_frame):
    # square contour order top right  ->  bottom right -> bottom left -> top left
    height, width = gray_frame.shape
    reference_pos = (width // 2, height // 2)
    contour_distance = []
    for c in contours:
        x,y,h,w = cv2.boundingRect(c)
        c_center = (x + (w // 2), y + (w // 2))
        contour_distance.append((c,calculate_distance(c_center, reference_pos) ))

    sorted_contours = sorted(contour_distance, key=lambda x: x[1])
    return  [ c[0] for c in sorted_contours[:9]]

def calculate_distance (p1,p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + ((p1[1] - p2[1]) ** 2))



def reduce_contour_complexity(contours):
    approx_contours = []

    for idx, c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.2 * peri, True)
        print(approx.shape)
        if (is_a_valid_contour(approx)):
            approx_contours.append(approx)

    return contours


def is_approximate(l1,l2):
    return l2 * 0.8 < l1 < l2 * 1.2

def is_a_valid_contour(contour):
    has_four_corners = len(contour) == 4
    if not has_four_corners:
        return False
    tr,br,bl,tl = contour

    tr,br,bl,tl = tr[0],br[0],bl[0],tl[0]
    line_lenght_1 = np.linalg.norm(tr-br)
    line_lenght_2 = np.linalg.norm(br-bl)
    line_lenght_3 = np.linalg.norm(bl-tl)
    line_lenght_4 = np.linalg.norm(tl-tr)

    """
            line 4
              -----
        line 3|   |  line 1
              -----
             line 2
    """
    are_lines_similar = \
            is_approximate(line_lenght_1, line_lenght_3) \
        and is_approximate(line_lenght_1, line_lenght_4) \
        and is_approximate(line_lenght_1, line_lenght_2) \
        and is_approximate(line_lenght_3, line_lenght_2) \
        and is_approximate(line_lenght_3, line_lenght_4) \
        and is_approximate(line_lenght_2, line_lenght_4) \
        and is_approximate(line_lenght_3, line_lenght_4) 
    
    """
cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle = np.arccos(cosine_angle)
    """
    angle_1 =np.degrees(np.arccos(np.dot(tl-tr,tr-br) / (np.linalg.norm(tl-tr)* np.linalg.norm(tr-br))))
    angle_2 =np.degrees(np.arccos(np.dot(tr-br,br-bl) / (np.linalg.norm(tr-br)* np.linalg.norm(br-bl))))
    angle_3 =np.degrees(np.arccos(np.dot(br-bl,bl-tl) / (np.linalg.norm(br-bl)* np.linalg.norm(bl-tl))))
    angle_4 =np.degrees(np.arccos(np.dot(bl-tl,tl-tr) / (np.linalg.norm(bl-tl)* np.linalg.norm(tl-tr))))
    
    are_angles_90_degrees = \
            is_approximate(angle_1, 90) and \
            is_approximate(angle_2, 90) and \
            is_approximate(angle_3, 90) and \
            is_approximate(angle_4, 90)

    return are_angles_90_degrees and are_lines_similar and has_four_corners


def find_face(contours, hierarchy):
    contours_child = {}

    for idx, h in enumerate(hierarchy[0]):
        _,_,first_child,parent = h  # Index of the first child
        if parent != -1:
            if parent not in contours_child:
                contours_child[parent] = set()
            contours_child[parent].add(first_child)
    
    print(contours_child)



if __name__ == "__main__":
    #cv2.namedWindow("Canny Lines", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Dilated Canny Lines", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("Detected Lines", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Image contours", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("HSV", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("image_ct", cv2.WINDOW_FREERATIO)

    cv2.namedWindow("video", cv2.WINDOW_FREERATIO)

    #capture = cv2.VideoCapture("./20241128_111931.mp4")
    #capture = cv2.VideoCapture("./20241123_182441.mp4")
    capture = cv2.VideoCapture(0)
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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = apply_canny_detection(original_frame)
        dilated_canny = dilate_contours(canny)

        #cv2.imshow( "Canny Lines",canny)
        cv2.imshow( "Dilated Canny Lines",dilated_canny)

        contours, hierarchy = cv2.findContours(gray_frame.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        contours = reduce_contour_complexity(contours)
        print(len(contours))
        for c in contours:
            cv2.drawContours(frame, c, -1, (0,0,255), thickness=3)
            cv2.imshow("video",frame)

        subsquares = detect_closest_nine_squares(contours, gray_frame)
        for c in subsquares:
            rect = cv2.boundingRect(c)
            cv2.rectangle(frame, rect, (0,255,0), thickness=4)
            cv2.imshow("video",frame)

