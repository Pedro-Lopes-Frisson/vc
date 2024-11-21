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

imgr1 = cv2.imread("../images/right01.jpg")

un_imgl1 = cv2.undistort(imgl1, mtx1,dist1)
cv2.imshow("Imgl1 OG", imgl1)
cv2.imshow("Imgl1 und", un_imgl1)

un_imgr1 = cv2.undistort(imgr1, mtx2,dist2)
cv2.imshow("Imgr1 OG", imgr1)
cv2.imshow("Imgr1 Und", un_imgr1)

cv2.waitKey()
cv2.destroyAllWindows()
