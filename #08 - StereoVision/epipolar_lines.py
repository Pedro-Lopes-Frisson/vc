import numpy as np
import cv2
import glob


with np.load("stereoParams.npz") as cam_params:
    mtx1 = cam_params["intrinsics1"]
    dist1 = cam_params["distortion1"]
    mtx2 = cam_params["intrinsics2"]
    dist2 = cam_params["distortion2"]
    R = cam_params["R"]
    T = cam_params["T"]
    E = cam_params["E"]
    F = cam_params["F"]

print(f"{mtx1=:}")
print(f"{dist1=:}")
print(f"{mtx2=:}")
print(f"{dist2=:}")
print(f"{R=:}")
print(f"{T=:}")
print(f"{E=:}")
print(f"{F=:}")


def mouse_handler_left(event, x, y, flags, params):
    global F
    global left_img_points
    if event == cv2.EVENT_LBUTTONDOWN:
        p = np.asarray([x, y])
        epilineR = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 1, F)
        epilineR = epilineR.reshape(-1, 3)[0]
        left_img_points.append(epilineR)


def mouse_handler_right(event, x, y, flags, params):
    global F
    global right_img_points
    if event == cv2.EVENT_LBUTTONDOWN:
        p = np.asarray([x, y])
        epilineR = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 0, F)
        epilineR = epilineR.reshape(-1, 3)[0]
        right_img_points.append(epilineR)


imgl1 = cv2.imread("../images/left01.jpg")
imgr1 = cv2.imread("../images/right01.jpg")

imgl1 = cv2.undistort(imgl1, mtx1, dist1)
cv2.imshow("Imgl1", imgl1)

imgr1 = cv2.undistort(imgr1, mtx2, dist2)
cv2.imshow("Imgr1", imgr1)

left_img_points = []
right_img_points = []

cv2.setMouseCallback("Imgl1", mouse_handler_left, left_img_points)
cv2.setMouseCallback("Imgr1", mouse_handler_right, right_img_points)

r,c = imgr1.shape[:2]
while True:
    # The function waitKey waits for a key event infinitely (when delay<=0)
    k = ""
    try:
        k = chr(cv2.waitKey(-1))
    except:
        pass

    if k == "q" or k == 27:  # toggle current image
        cv2.destroyAllWindows()
        break
    elif k == "c":
        for r in left_img_points:
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(imgr1, (x0,y0), (x1,y1), color,1)
        for r in right_img_points:
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(imgl1, (x0,y0), (x1,y1), color,1)

    cv2.imshow("Imgr1", imgr1)
    cv2.imshow("Imgl1", imgl1)
