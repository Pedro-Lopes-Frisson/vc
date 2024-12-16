import cv2
import numpy as np
import math
import sys


def apply_canny_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.bilateralFilter(gray, 9, 15,35)
    cv2.imshow("Blurred", blurred)
    dst = cv2.Canny(blurred, 10,30 , None)
    cv2.imshow("Canny", dst)
    return dst


def dilate_contours(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    dilated = cv2.dilate(image, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    eroded = cv2.bitwise_not(eroded)
    cv2.imshow("eroded", eroded)
    return eroded

def get_hough_lines(image):
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 50, None, 80, 8)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


def remove_larger_nested_squares(contours):
    filtered_contours = []
    for i, c1 in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        is_larger = False
        for j, c2 in enumerate(contours):
            if i != j:
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                # Check if c2 (smaller) is inside c1 (larger)
                if x2 >= x1 and y2 >= y1 and x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1:
                    if cv2.contourArea(c1) > cv2.contourArea(c2):  # Compare areas
                        is_larger = True
                        break
        if not is_larger:
            filtered_contours.append(c1)
    return filtered_contours

def reduce_contour_complexity(image,frame):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    image_ct = frame.copy()

    last_child_contours = contours
    #last_child_contours = find_last_child_contours(contours, hierarchy)
    #last_child_contours = remove_larger_nested_squares(contours)

    for c in last_child_contours:
        area = cv2.contourArea(c)
        if area < 800:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        #print(f"contours before {len (c)} after {len(approx)}")
        approx_contours.append(approx)
    
    return last_child_contours,approx_contours

def is_approximate(l1,l2):
    return l2 * 0.7 < l1 < l2 * 1.3

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
    biggest_line = max(line_lenght_1,line_lenght_2,line_lenght_3,line_lenght_4)

    """
            line 4
              -----
        line 3|   |  line 1
              -----
             line 2
    """
    are_lines_similar = \
            is_approximate(biggest_line, line_lenght_1) \
        and is_approximate(biggest_line, line_lenght_2) \
        and is_approximate(biggest_line, line_lenght_3) \
        and is_approximate(biggest_line, line_lenght_4) 

    print("ARE LINES SIMILAR ", are_lines_similar, "\n\n\n\n")
    print(f"{biggest_line=:}")
    print(f"{line_lenght_1=:}")
    print(f"{line_lenght_2=:}")
    print(f"{line_lenght_3=:}")
    print(f"{line_lenght_4=:}")
    
    """
cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle = np.arccos(cosine_angle)
    """
    angle_1 =np.degrees(np.arccos(np.dot(tl-tr,tr-br) / (np.linalg.norm(tl-tr)* np.linalg.norm(tr-br))))
    angle_2 =np.degrees(np.arccos(np.dot(tr-br,br-bl) / (np.linalg.norm(tr-br)* np.linalg.norm(br-bl))))
    angle_3 =np.degrees(np.arccos(np.dot(br-bl,bl-tl) / (np.linalg.norm(br-bl)* np.linalg.norm(bl-tl))))
    angle_4 =np.degrees(np.arccos(np.dot(bl-tl,tl-tr) / (np.linalg.norm(bl-tl)* np.linalg.norm(tl-tr))))


    print("ARE LINES SIMILAR ", angle_1, "\n\n\n\n")
    print("ARE LINES SIMILAR ", angle_2, "\n\n\n\n")
    print("ARE LINES SIMILAR ", angle_3, "\n\n\n\n")
    print("ARE LINES SIMILAR ", angle_4, "\n\n\n\n")
    are_angles_90_degrees = \
            is_approximate(angle_1, 90) and \
            is_approximate(angle_2, 90) and \
            is_approximate(angle_3, 90) and \
            is_approximate(angle_4, 90)

    #print("ARE LINES SIMILAR ", are_angles_90_degrees, "\n\n\n\n")
    return are_angles_90_degrees and are_lines_similar and has_four_corners

def detect_middle_square(contours):
    """ o quadrado do meio e o que vai ter menor distacia do resto dos quadraos excluindo ele proprio"""

    contour_distance = []
    for idx,c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        c_center = (x + (w // 2), y + (w // 2))
        max_distance = 0
        for idx1,c1 in enumerate(contours):
            if idx == idx1:
                continue

            x1,y1,w1,h1 = cv2.boundingRect(c1)
            c_center1 = (x1 + (w1 // 2), y1 + (w1 // 2))
            distance = calculate_distance(c_center, c_center1)
            if max_distance <  distance:
                max_distance = distance

        contour_distance.append((c, max_distance ))

    sorted_contours = sorted(contour_distance, key=lambda x: x[1])
    #print([ c[0] for c in sorted_contours[:1]])
    return  [ c[0] for c in sorted_contours[:1]]

def calculate_distance (p1,p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + ((p1[1] - p2[1]) ** 2))


def order_points(pts):
    # Ensure pts is a numpy array
    pts = np.array(pts.reshape((4,2)))
    print(pts.shape)
    print(f"{pts=:}")

    # Validate the shape of pts
    if pts.shape[0] != 4 or pts.shape[1] != 2:
        raise ValueError("Input pts must be a 4x2 array.")

    rect = np.zeros((4, 2), dtype="float32")

    # Calculate the sum and difference for sorting
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Assign corners of the rectangle
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def get_center(contour):
    moments = cv2.moments(contour)
    return (moments["m10"]//moments["m00"],moments["m01"]//moments["m00"] )

def is_row_neighbor(contour1, contour2):
    center_1 = get_center(contour1)
    center_2 = get_center(contour2)

    if is_approximate(center_1[0], center_2[0]):
        return True
    return False

def is_col_neighbor(contour1, contour2):
    center_1 = get_center(contour1)
    center_2 = get_center(contour2)

    if is_approximate(center_1[1], center_2[1]):
        return True
    return False


if __name__ == "__main__":
    #cv2.namedWindow("Canny Lines", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("Dilated Canny Lines", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("Detected Lines", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("Image contours", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("HSV", cv2.WINDOW_FREERATIO)
    #cv2.namedWindow("image_ct", cv2.WINDOW_FREERATIO)

    cv2.namedWindow("video", cv2.WINDOW_FREERATIO)

    capture = cv2.VideoCapture("./20241128_111931.mp4")
    #capture = cv2.VideoCapture("./20241123_182441.mp4")
    #capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture("http://192.168.241.75:4747/video")
    #capture = cv2.VideoCapture("http://192.168.1.68:4747/video")

    middle_square_d = None
    square_size = 600
    dst_points = np.array([
        [0, 0],
        [square_size - 1, 0],
        [square_size - 1, square_size - 1],
        [0, square_size - 1]
    ], dtype="float32")

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

        #print("\n\nNEW FRAME\n\n")
        original_frame = frame.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = frame.shape[:2]
        image_center = (height // 2, width // 2 )

        canny = apply_canny_detection(frame)
        dilated_canny = dilate_contours(canny)
        contours, approx_contours = reduce_contour_complexity(dilated_canny, frame)

        """
        middle_square_d = detect_middle_square(approx_contours)
        ordered_points = order_points(middle_square_d)

        matrix = cv2.getPerspectiveTransform(ordered_points, dst_points)
        warped = cv2.warpPerspective(frame, matrix, frame.shape[:2])
        cv2.imshow("Warped", warped)


        #for i in contours:
            #cv2.drawContours(frame, i, -1, (255,0,0),thickness=2)
            
        """
        valid_contours = []
        
        for i in approx_contours:
            x1, y1, w1, h1 = cv2.boundingRect(i)
            hw, hh = (w1//4, h1//4)
            rect = (x1+hh,y1+hw,w1//2,h1//2)
            area = cv2.contourArea(i)
            if is_a_valid_contour(i) and  area < (frame.shape[0] * frame.shape[1])// 4:
                #cv2.rectangle(frame, rect, (0,255,0),thickness=2)
                valid_contours.append(order_points(i))
            """
            else:
                if len(valid_contours)< 9:
                    cv2.rectangle(frame, rect, (255,0,0),thickness=2)
                    cv2.waitKey()
            """
            cv2.imshow("video", frame)

        neighboors_dict = {}
        for c in valid_contours:
            for c1 in valid_contours:
                key = tuple(c.flatten())
                if  key not in neighboors_dict:
                    neighboors_dict[key] = {"col": [], "row":[]}

                if is_row_neighbor(c,c1):
                    neighboors_dict[key]["row"].append(c1)
                if is_col_neighbor(c,c1):
                    neighboors_dict[key]["col"].append(c1)

        for k, v in neighboors_dict.items():
            c = np.asarray(k).reshape((1,4,2))
            x1, y1, w1, h1 = cv2.boundingRect(c)
            hw, hh = (w1//4, h1//4)
            rect = (x1+hh,y1+hw,w1//2,h1//2)
            cv2.rectangle(frame, rect, (255,255,0),thickness=2)
            for c in v["col"]:
                x1, y1, w1, h1 = cv2.boundingRect(c)
                hw, hh = (w1//4, h1//4)
                rect = (x1+hh,y1+hw,w1//2,h1//2)
                cv2.rectangle(frame, rect, (0,255,0),thickness=2)
            for r in v["row"]:
                x1, y1, w1, h1 = cv2.boundingRect(r)
                hw, hh = (w1//4, h1//4)
                rect = (x1+hh,y1+hw,w1//2,h1//2)
                cv2.rectangle(frame, rect, (0,0,255),thickness=2)
            cv2.waitKey()
            cv2.imshow("video",frame )





