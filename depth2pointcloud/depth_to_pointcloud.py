import argparse
import sys
import os
from PIL import Image
import numpy as np
import math
import cv2
import scipy.misc
import matplotlib.pyplot as plt
import tifffile as tiff


if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print('set parameters.')
        print('python depth_to_pointcloud.py depthMapFile pointCloudFile group rgbFile')
        sys.exit()

    depthMapFileName   = sys.argv[1]
    pointCloudFileName = sys.argv[2]
    group              = int(sys.argv[3])
    rgbFileName        = sys.argv[4]
    
    # camera parameters
    camera_matrices = [None,None]
    distortions = [None,None]
    rotations = [None,None]
    translations = [None,None]

    W = 1280
    H = 1024
    X_Border = 0
    Y_Border = 0

    if(group == 1):
        ### dataset_8
        print("Group = {}.".format(group))
        camera_matrices[0] = np.array((
            (1.02408777e+03, 0., 6.01806702e+02),
            (0., 1.02389478e+03, 5.08131683e+02), 
            (0., 0., 1.)), dtype = np.float64)
        distortions[0] = np.array(
            (-2.52257544e-03, 4.38565714e-03, 0., 0., 9.21500759e-05), dtype = np.float64)
        camera_matrices[1] = np.array((
            (1.02419836e+03, 0., 6.96750427e+02), 
            (0., 1.02398749e+03, 5.07494263e+02),
            (0., 0., 1.)), dtype = np.float64)
        distortions[1] = np.array(
            (-3.26306466e-03, 5.70008671e-03, 0., 0., 7.57322850e-05), dtype = np.float64)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float64)
        translations[0] = np.array((0,0,0), dtype = np.float64)
        rotations[1] = np.array((
            (9.99999404e-01, -1.24090354e-06, -1.11142127e-03),
            (1.29263049e-06, 1., 4.65405355e-05),
            (1.11142127e-03, -4.65419434e-05, 9.99999404e-01)), dtype = np.float64)
        translations[1] = np.array(
            (-4.34818459e+00, 2.83603016e-02, -9.00963729e-04), dtype = np.float64)

    elif(group == 2):
        ### dataset_9
        print("Group = {}.".format(group))
        camera_matrices[0] = np.array((
            (1.02342310e+03, 0., 5.95927734e+02),
            (0., 1.02322693e+03, 5.10647095e+02), 
            (0., 0., 1.)), dtype = np.float)
        distortions[0] = np.array(
            (1.98164504e-04, -2.01636995e-03, 0., 0., 2.62780651e-03), dtype = np.float)
        camera_matrices[1] = np.array((
            (1.02330627e+03, 0., 6.85855286e+02), 
            (0., 1.02314386e+03, 5.11527374e+02),
            (0., 0., 1.)), dtype = np.float)
        distortions[1] = np.array(
            (4.21567587e-04, -2.37609493e-03, 0., 0., 2.29821727e-03), dtype = np.float)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float)
        translations[0] = np.array((0,0,0), dtype = np.float)
        rotations[1] = np.array((
            (1., 1.89964849e-05, -1.73123670e-04), 
            (-1.89916173e-05, 1., 2.81125340e-05), 
            (1.73124194e-04, -2.81092452e-05, 1.)), dtype = np.float)
        translations[1] = np.array(
            (-4.13235331e+00, -4.20858636e-02, -2.61657196e-03), dtype = np.float)


    else:
        print('parameter error')
        sys.exit()
    
    #left camera
    fx = camera_matrices[0][0][0]
    fy = camera_matrices[0][1][1]
    cx = camera_matrices[0][0][2]
    cy = camera_matrices[0][1][2] 
    print("fx={}; fy={}; cx={}; cy={}".format(fx, fy, cx, cy))

    rgbImage       = cv2.imread(rgbFileName)
    depthMap       = tiff.imread(depthMapFileName)
    pointCloudFile = open(pointCloudFileName, 'w')

    for v in range(H): 
        for u in range(W): 
            b, g, r = rgbImage[v, u]
            Z = depthMap[v,u]  #####################################################
            X = (u - cx) * Z / fx 
            Y = (v - cy) * Z / fy 
            Z =  Z * 1000
            X = X * 1000
            Y = Y * 1000
            pointCloudFile.write('{} {} {} {} {} {}\n'.format(X, Y, Z, r, g, b))

    pointCloudFile.close()
    print("done.")