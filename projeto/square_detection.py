import cv2
import numpy as np
import math
import sys

def apply_canny_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 15,35)
    cv2.imshow("Blurred", blurred)
    dst = cv2.Canny(blurred, 10,30 , None)
    cv2.imshow("Canny", dst)
    return dst

def dilate_contours(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11,11))
    dilated = cv2.dilate(image, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def get_hough_lines(image):
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 50, None, 80, 8)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

def calculate_distance (p1,p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + ((p1[1] - p2[1]) ** 2))

def detect_closest_nine_squares(contours,reference_square, gray_frame):
    # square contour order top right  ->  bottom right -> bottom left -> top left
    reference_pos = reference_square

    contour_distance = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_center = (x + (w // 2), y + (h // 2))
        contour_distance.append((c,calculate_distance(c_center, reference_pos) ))

    sorted_contours = sorted(contour_distance, key=lambda x: x[1])
    return  [ c[0] for c in sorted_contours[:9]]

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
    

def reduce_contour_complexity(image,frame):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    image_ct = frame.copy()

    #last_child_contours = contours
    #last_child_contours = find_last_child_contours(contours, hierarchy)
    last_child_contours = remove_larger_nested_squares(contours)

    for c in last_child_contours:

        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        #print(f"contours before {len (c)} after {len(approx)}")
        if is_a_valid_contour(approx):
            approx_contours.append(approx)
    
    return last_child_contours,approx_contours

def detect_contout_main_color(contours, frame_hsv):
    green_lower=np.array([54,63,79])
    green_upper=np.array([68,222,157])

    yellow_lower=np.array([22,106,178])
    yellow_upper=np.array([39,222,245])

    red_lower=np.array([0,101,25])
    red_upper=np.array([5,255,207])

    blue_lower=np.array([88,146,0])
    blue_upper=np.array([109,239,186])

    orange_lower=np.array([6,162,175])
    orange_upper=np.array([17,255,255])

    white_lower=np.array([18,0,216])
    white_upper=np.array([40,39,255])
    

    for c in contours:
        contour_middle_x, contour_middle_y = math.floor(c[3][0][0] + ((c[0][0][0] - c[3][0][0] )/ 2)),math.floor(c[0][0][1] + ((c[1][0][1] - c[0][0][1] )/ 2))
        color = np.zeros_like( frame_hsv[0,0])
        #print(color)

        contour_middle_x = min(max(contour_middle_x, 0), frame_hsv.shape[0] - 1)
        contour_middle_y = min(max(contour_middle_y, 0), frame_hsv.shape[1] - 1)
        for point in c:
            x,y = point[0]
            x = min(max(x, 0), frame_hsv.shape[0] - 1)
            y = min(max(y, 0), frame_hsv.shape[1] - 1)
            #print(x,y,frame_hsv.shape[1], frame_hsv.shape[0])
            color += frame_hsv[x,y]

        color += frame_hsv[contour_middle_x,contour_middle_y]
        if all( color > green_lower) and all( color < green_upper):
            cv2.drawContours(frame_hsv,c, -1, (255,255,0))
            cv2.circle(frame_hsv, (contour_middle_x, contour_middle_y), 10, (100,10,100), 5)
            cv2.putText(frame_hsv, "green",(contour_middle_x, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.putText(frame_hsv, f"green {color}{green_lower}{green_upper}",(contour_middle_x, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=5)
            cv2.imshow("video", frame_hsv)
            

        elif all( color > red_lower) and all( color < red_upper):
            #print( f" {c[3][0][0]} + {((c[0][0][0] - c[3][0][0] )/ 2) },{c[0][0][1]} + {((c[1][0][1] - c[0][0][1] )/ 2)}")
            cv2.drawContours(frame_hsv,c, -1, (255,255,0))
            cv2.circle(frame_hsv, (contour_middle_x, contour_middle_y), 10, (100,10,100), 5)
            cv2.putText(frame_hsv, f"red {color=:}{red_lower=:}{red_upper=:}",(contour_middle_x-100, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.imshow("video", frame_hsv)

        elif all( color > yellow_lower) and all( color < yellow_upper):
            cv2.drawContours(frame_hsv,c, -1, (255,255,0))
            cv2.circle(frame_hsv, (contour_middle_x, contour_middle_y), 10, (100,10,100), 5)
            cv2.putText(frame_hsv, "yellow",(contour_middle_x, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.imshow("video", frame_hsv)

        elif all( color > blue_lower) and all( color < blue_upper):
            #print( f" {c[3][0][0]} + {((c[0][0][0] - c[3][0][0] )/ 2) },{c[0][0][1]} + {((c[1][0][1] - c[0][0][1] )/ 2)}")
            cv2.drawContours(frame_hsv,c, -1, (255,255,0))
            cv2.circle(frame_hsv, (contour_middle_x, contour_middle_y), 10, (100,10,100), 5)
            cv2.putText(frame_hsv, "blue",(contour_middle_x, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.imshow("video", frame_hsv)

        elif all( color > orange_lower) and all( color < orange_upper):
            cv2.drawContours(frame_hsv,c, -1, (255,255,0))
            cv2.circle(frame_hsv, (contour_middle_x, contour_middle_y), 10, (100,10,100), 5)
            cv2.putText(frame_hsv, "orange",(contour_middle_x, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.imshow("video", frame_hsv)

        elif all( color > white_lower) and all( color < white_upper):
            cv2.drawContours(frame_hsv,c, -1, (255,255,0))
            cv2.circle(frame_hsv, (contour_middle_x, contour_middle_y), 10, (100,10,100), 5)
            cv2.putText(frame_hsv, "white",(contour_middle_x, contour_middle_y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            cv2.imshow("video", frame_hsv)

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
    
    #print("ARE LINES SIMILAR ", angle_1, "\n\n\n\n")
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

def detect_face_color(middle_square, hsv_frame):

    x_r,y_r,width, height = cv2.boundingRect(middle_square[0])
    
    reference_pos_x, reference_pos_y = (x_r + (width // 2), y_r + (height // 2))
    hsv_px = hsv_frame[reference_pos_y,reference_pos_x]
    if 0 <= hsv_px[1] <  35 :
        return "White"

    if 0< hsv_px[0] < 7 or 170 < hsv_px[0] <= 179:
        return "RED"

    if 7 < hsv_px[0] < 20:
        return "Orange"

    if 20< hsv_px[0] < 55:
        return "Yellow"

    if 55 < hsv_px[0] < 85:
        return "Green"

    if 85< hsv_px[0] < 115:
        return "Blue"
    return " "

def get_face_string(square, hsv_frame):
    for i in sorted_contours:
        print(detect_face_color(i, hsv_frame)[0], end="")

def get_center(contour):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return(cx,cy)

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
    #capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture("http://192.168.241.75:4747/video")
    capture = cv2.VideoCapture("http://192.168.1.68:4747/video")

    middle_square_d = None

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
        lines = cv2.HoughLinesP(dilated_canny, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow( "Image contours",frame)

        #cv2.imshow( "Canny Lines",canny)
        #cv2.imshow( "Dilated Canny Lines",dilated_canny)
        contours, approx_contours = reduce_contour_complexity(dilated_canny, frame)

        """
        subsquares = detect_closest_nine_squares(contours, image_center, gray_frame)

        middle_square = detect_middle_square(contours)
        if middle_square != []:
            middle_square_d = middle_square

        if not middle_square_d:
            continue

        color_name = detect_face_color(middle_square_d,hsv_frame)

        cv2.putText(frame, color_name, (10,500) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)

        sorted_contours = sorted(subsquares, key= lambda x: get_center(x)[::-1],reverse=True)

        cv2.drawContours(frame, middle_square_d, -1, (129,10,255), thickness=3)
        cv2.imshow("video",frame)
        face_string = get_face_string(sorted_contours, hsv_frame)
        print(face_string)
        """
            

        for idx ,approx in enumerate(contours):
            #cv2.drawContours(frame, approx, -1, (255,0,0), thickness= 2)
            """
            area = cv2.contourArea(approx)
            rect = cv2.boundingRect(approx)
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + int(w * 0.9), y +int(h * 0.9)), (255, 0, 255), 2)  # Purple color
            for point in approx:
                approx_x,approx_y = point[0]
                cv2.circle(frame, (approx_x,approx_y), 2, (255,255,255), thickness=2)
            """
            cv2.drawContours(frame, approx, -1, (255,0,0), 2)
            #cv2.putText(frame, f"{area}", (int(x+(w/2)), int(y+(h/2))), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
            #cv2.putText(frame, f"{idx}", (int(x+(w/2)), int(y+(h/2) - 12)), cv2.FONT_HERSHEY_PLAIN,fontScale=1, lineType=cv2.LINE_AA, thickness=1, color=(255,255,255))
            cv2.imshow( "Image contours",frame)
            #cv2.imshow("HSV", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))


        for idx ,approx in enumerate(approx_contours):
            cv2.drawContours(frame, [approx], -1, (0,255,0), 2)
            cv2.imshow( "Image contours",frame)
