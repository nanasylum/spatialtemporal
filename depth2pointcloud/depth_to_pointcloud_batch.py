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
        print('python depth_to_pointcloud.py depthMapFolder pointCloudFolder group rgbFolder')
        sys.exit()

    depthMapFolder   = sys.argv[1]
    pointCloudFolder = sys.argv[2]
    group            = int(sys.argv[3])
    rgbFolder        = sys.argv[4]
    
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
            (0., 0., 1.)), dtype = np.float64)
        distortions[0] = np.array(
            (1.98164504e-04, -2.01636995e-03, 0., 0., 2.62780651e-03), dtype = np.float64)
        camera_matrices[1] = np.array((
            (1.02330627e+03, 0., 6.85855286e+02), 
            (0., 1.02314386e+03, 5.11527374e+02),
            (0., 0., 1.)), dtype = np.float64)
        distortions[1] = np.array(
            (4.21567587e-04, -2.37609493e-03, 0., 0., 2.29821727e-03), dtype = np.float64)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float64)
        translations[0] = np.array((0,0,0), dtype = np.float64)
        rotations[1] = np.array((
            (1., 1.89964849e-05, -1.73123670e-04), 
            (-1.89916173e-05, 1., 2.81125340e-05), 
            (1.73124194e-04, -2.81092452e-05, 1.)), dtype = np.float64)
        translations[1] = np.array(
            (-4.13235331e+00, -4.20858636e-02, -2.61657196e-03), dtype = np.float64)
    elif(group == 3):
        ### dataset_1 ~ dataset_3
        print("Group = {}.".format(group))
        camera_matrices[0] = np.array((
            (1.03530811e+03, 0., 5.96955017e+02),
            (0., 1.03508765e+03,5.20410034e+02), 
            (0., 0., 1.)), dtype = np.float64)
        distortions[0] = np.array(
            (-5.95157442e-04, -5.46629308e-04, 0., 0., 1.82959007e-03), dtype = np.float64)
        camera_matrices[1] = np.array((
            (1.03517419e+03, 0., 6.88361877e+02), 
            (0., 1.03497900e+03, 5.21070801e+02),
            (0., 0., 1.)), dtype = np.float64)
        distortions[1] = np.array(
            (-2.34280655e-04, -7.68933969e-04, 0., 0., 7.76395318e-04), dtype = np.float64)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float64)
        translations[0] = np.array((0,0,0), dtype = np.float64)
        rotations[1] = np.array((
            (1., 1.94856493e-05, -1.52324792e-04), 
            (-1.95053162e-05, 1., -1.29114138e-04), 
            (1.52322275e-04, 1.29117107e-04, 1.)), dtype = np.float64)
        translations[1] = np.array(
            (-4.14339018e+00, -2.38197036e-02, -1.90685259e-03), dtype = np.float64)
    elif(group == 4):
    ### dataset_4 ~ dataset_5
        print("Group = {}.".format(group))
        camera_matrices[0] = np.array((
            (1.07364844e+03, 0., 5.77499695e+02),
            (0., 1.07343433e+03,5.24409790e+02), 
            (0., 0., 1.)), dtype = np.float64)
        distortions[0] = np.array(
            (-7.47532817e-04, 1.77790504e-03, 1.39573240e-04, 0., 0.), dtype = np.float64)
        camera_matrices[1] = np.array((
            (1.07266870e+03, 0., 6.76994690e+02), 
            (0., 1.07244336e+03, 5.23896667e+02),
            (0., 0., 1.)), dtype = np.float64)
        distortions[1] = np.array(
            (-1.31973054e-03, 3.40759335e-03, 1.06380983e-04, 0., 0.), dtype = np.float64)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float64)
        translations[0] = np.array((0,0,0), dtype = np.float64)
        rotations[1] = np.array((
            (9.99999344e-01, 9.92073183e-06, -1.13043236e-03), 
            (-9.90405715e-06, 1., 1.47561313e-05), 
            (1.13043259e-03, -1.47449255e-05, 9.99999344e-01)), dtype = np.float64)
        translations[1] = np.array(
            (-4.36337757e+00, 2.11443733e-02, -4.78740744e-02), dtype = np.float64)

    elif(group == 5):
        ### dataset_6 ~ dataset_7
        print("Group = {}.".format(group))
        camera_matrices[0] = np.array((
            (1.08697437e+03, 0., 5.86080322e+02),
            (0., 1.08676831e+03, 5.12475891e+02), 
            (0., 0., 1.)), dtype = np.float64)
        distortions[0] = np.array(
            (-1.59616291e-03, 4.46009031e-03, 1.10806825e-04, 0., 0.), dtype = np.float64)
        camera_matrices[1] = np.array((
            (1.08695398e+03, 0., 6.86717102e+02), 
            (0., 1.08677051e+03, 5.11879486e+02),
            (0., 0., 1.)), dtype = np.float64)
        distortions[1] = np.array(
            (-1.11811305e-03, 3.16335540e-03, 8.72377423e-05, 0., 0.), dtype = np.float64)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float64)
        translations[0] = np.array((0,0,0), dtype = np.float64)
        rotations[1] = np.array((
            (9.99999285e-01, -2.10470330e-06, -1.18983735e-03),
            (2.19584786e-06, 1., 7.66011581e-05),
            (1.18983723e-03, -7.66037119e-05, 9.99999285e-01)), dtype = np.float64)
        translations[1] = np.array(
            (-4.36411285e+00, 2.14479752e-02, -3.81391775e-03), dtype = np.float64)


    else:
        print('parameter error')
        sys.exit()

    #left camera
    fx = camera_matrices[0][0][0]
    fy = camera_matrices[0][1][1]
    cx = camera_matrices[0][0][2]
    cy = camera_matrices[0][1][2] 
    print("fx={}; fy={}; cx={}; cy={}".format(fx, fy, cx, cy))

    # keyframes = [
    #     "keyframe_0",
    #     "keyframe_1",
    #     "keyframe_2",
    #     "keyframe_3",
    #     "keyframe_4"
    # ]
    keyframes = [
        "keyframe1",
        "keyframe2",
        "keyframe3",
        "keyframe4",
        "keyframe5"
    ]

    for K in range(5):
        keyframeFolder = os.path.join(depthMapFolder, keyframes[K])
        framesName = os.listdir(keyframeFolder)
        framesName.sort()
        framesNum = len(framesName)

        pointCloudFolderName = os.path.join(pointCloudFolder, keyframes[K], "image_02","pc_rgb_swindepth")
        if not os.path.exists(pointCloudFolderName):
            os.mkdir(pointCloudFolderName)
        rgbImageFolder = os.path.join(rgbFolder, keyframes[K], "image_02", "data")

        depthMapName = os.path.join(keyframeFolder,  "image_02", "depthmap_swindepth", "frame000001.tiff")
        pointCloudName = os.path.join(pointCloudFolderName, "frame000001.txt")
        rgbImageName = os.path.join(rgbImageFolder, "0000000001.png")

        if not os.path.exists(depthMapName):
            continue
        rgbImage       = cv2.imread(rgbImageName)
        depthMap       = tiff.imread(depthMapName)
        pointCloudFile = open(pointCloudName, 'w')

        for v in range(H): 
                for u in range(W): 
                    b, g, r = rgbImage[v, u]
                    Z = depthMap[v,u]  #####################################################
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    Z =  Z * 1000
                    X = X * 1000
                    Y = Y * 1000
                    if(Z > 500):
                        continue
                    pointCloudFile.write('{} {} {} {} {} {}\n'.format(X, Y, Z, r, g, b))

        pointCloudFile.close()
        print("{} is done.".format(pointCloudFolderName))

        # for n in range(0,framesNum,24):
        #     print(framesName)
        #     depthMapName = os.path.join(keyframeFolder,  "image_02", "depthmap_ours", "frame000001.tiff")
        #     pointCloudName = os.path.join(pointCloudFolderName, framesName[n][0:-5]+".txt")
        #     rgbImageName = os.path.join(rgbImageFolder, "%010d.png"%n)
        #     print(depthMapName)
        #     print(pointCloudName)
        #     rgbImage       = cv2.imread(rgbImageName)
        #     depthMap       = tiff.imread(depthMapName)
        #     pointCloudFile = open(pointCloudName, 'w')

        #     for v in range(H): 
        #         for u in range(W): 
        #             b, g, r = rgbImage[v, u]
        #             Z = depthMap[v,u]  #####################################################
        #             X = (u - cx) * Z / fx
        #             Y = (v - cy) * Z / fy
        #             Z =  Z * 1000
        #             X = X * 1000
        #             Y = Y * 1000
        #             pointCloudFile.write('{} {} {} {} {} {}\n'.format(X, Y, Z, r, g, b))

        #     pointCloudFile.close()
        #     print("{} is done.".format(framesName[n]))