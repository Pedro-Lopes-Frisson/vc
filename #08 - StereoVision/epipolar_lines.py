import numpy as np
import cv2
import glob

def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("left click")




import numpy as np
import cv2
import glob

with np.load('stereoParams.npz') as cam_params:
    mtx1=cam_params['intrinsics1']
    dist1=cam_params['distortion1']
    mtx2=cam_params['intrinsics2']
    dist2=cam_params['distortion2']
    R=cam_params['R']
    T=cam_params['T']
    E=cam_params['E']
    F=cam_params['F']

print(f"{mtx1=:}")
print(f"{dist1=:}")
print(f"{mtx2=:}")
print(f"{dist2=:}")
print(f"{R=:}")
print(f"{T=:}")
print(f"{E=:}")
print(f"{F=:}")



imgl1 = cv2.imread("../images/left01.jpg")
imgl2 = cv2.imread("../images/left01.jpg")


imgr1 = cv2.imread("../images/right01.jpg")
imgr2 = cv2.imread("../images/right01.jpg")


imgl1 = cv2.undistort(imgl1, mtx1,dist1)
cv2.imshow(imgl1, "Imgl1")

imgr1 = cv2.undistort(imgr1, mtx2,dist2)
cv2.imshow(imgr1, "Imgr1")



cv2.setMouseCallback("Window", mouse_handler)
