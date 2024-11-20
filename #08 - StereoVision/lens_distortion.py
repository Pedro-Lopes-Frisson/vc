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


