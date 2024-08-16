# coding=utf-8
###############################################################################
# 输入：
#     disparity_folder: 预测的视差图(*.png格式)文件夹路径
#     save_folder: 点云文件(*.txt格式)保存的文件夹路径
#     Group_num: 指定数据的组数，取值为：1,2,3
#
# 用法: 
#     python convert_disparity_image_to_pointcloud.py disparity_folder save_folder Group
#
# @Huoling Luo
#
###############################################################################

import numpy as np
import math
import os
import sys
import cv2
import scipy.misc
import matplotlib.pyplot as plt

"""
There are 7 datasets in MICCAI 2019 SCARED from three different pigs, each dataset
has different camera parameters, they can be divided different three groups by camera
parameters as:

Group 1: dataset_1 ~ dataset_3
Group 2: dataset_4 ~ dataset_5
Group 3: dataset_6 ~ dataset_7

So, we need specify the group number to load different camera parameters when evaluation.
"""

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('set parameters.\r\npython disparity_file result.txt Group')
        sys.exit()

    # camera parameters
    camera_matrices = [None,None]
    distortions = [None,None]
    rotations = [None,None]
    translations = [None,None]

    W = 1280
    H = 1024
    X_Border = 100
    Y_Border = 0

    group = sys.argv[3]
    print("group = {}".format(group))
    if(group == '1'):
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

    elif(group == '2'):
        ### dataset_4 ~ dataset_5
        print("Group = {}.".format(group))
        camera_matrices[0] = np.array((
            (1.07364844e+03, 0., 5.77499695e+02),
            (0., 1.07343433e+03,5.24409790e+02), 
            (0., 0., 1.)), dtype = np.float32)
        distortions[0] = np.array(
            (-7.47532817e-04, 1.77790504e-03, 1.39573240e-04, 0., 0.), dtype = np.float32)
        camera_matrices[1] = np.array((
            (1.07266870e+03, 0., 6.76994690e+02), 
            (0., 1.07244336e+03, 5.23896667e+02),
            (0., 0., 1.)), dtype = np.float32)
        distortions[1] = np.array(
            (-1.31973054e-03, 3.40759335e-03, 1.06380983e-04, 0., 0.), dtype = np.float32)
        rotations[0] = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)), dtype = np.float32)
        translations[0] = np.array((0,0,0), dtype = np.float32)
        rotations[1] = np.array((
            (9.99999344e-01, 9.92073183e-06, -1.13043236e-03), 
            (-9.90405715e-06, 1., 1.47561313e-05), 
            (1.13043259e-03, -1.47449255e-05, 9.99999344e-01)), dtype = np.float32)
        translations[1] = np.array(
            (-4.36337757e+00, 2.11443733e-02, -4.78740744e-02), dtype = np.float32)

    elif(group == '3'):
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



    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1= camera_matrices[0],
    distCoeffs1= distortions[0],
    cameraMatrix2 = camera_matrices[1],
    distCoeffs2=distortions[1],
    imageSize=(W,H),
    R=rotations[1],
    T=translations[1],
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.0)
    #print(Q)

    rectification_maps = [[None,None],[None,None]]
    rectification_maps[0][0], rectification_maps[0][1] = \
    cv2.initUndistortRectifyMap(
        cameraMatrix=camera_matrices[0],
        distCoeffs=distortions[0],
        R=R1,
        newCameraMatrix=P1,
        size=(W,H),
        m1type=cv2.CV_32FC2
    )
    rectification_maps[1][0], rectification_maps[1][1] = \
    cv2.initUndistortRectifyMap(
        cameraMatrix=camera_matrices[1],
        distCoeffs=distortions[1],
        R=R2,
        newCameraMatrix=P2,
        size=(W,H),
        m1type=cv2.CV_32FC2
    )
    
    cam_to_world = np.linalg.inv(R1)

    disparity_folder = sys.argv[1]
    save_folder = sys.argv[2]
    
    names = os.listdir(disparity_folder)
    names.sort()
    num = len(names)
    print(names)
    print(num)
    bad_n = 0
    
    for n in range(num):  
        disparity_path = os.path.join(disparity_folder, names[n])
        print(disparity_path)
        
        out_name = os.path.join(save_folder, names[n])[0:-4] + '.txt'
        save_file = open(out_name, 'w')     
        print(out_name)  

        # predicted file is png format, should be convert to point cloud first
        import imageio
        disp_map = imageio.imread(disparity_path, mode="F")
        # disp_map = scipy.misc.imread(disparity_path, mode="F")
        points = cv2.reprojectImageTo3D(disparity=disp_map, Q=Q)      
        print(n)
    
        for i in range(H):
            for j in range(W):
                ux, uy = rectification_maps[0][0][i][j]
                if(ux < X_Border or ux > points.shape[1]):
                    continue
                if(uy < Y_Border or uy > points.shape[0] - Y_Border):
                    continue

                point = points[i][j]
                point = np.matmul(cam_to_world,point)

                pred_x = point[0] #* 1.9
                pred_y = point[1] #* 1.9
                pred_z = point[2] #* 1.9

                """            
                # 第一组数据预测的深度值有个偏移
                if group == 1:
                    pred_x = point[0] * 2.3
                    pred_y = point[1] * 2.3
                    pred_z = point[2] * 2.3
                else:
                    pred_x = point[0]
                    pred_y = point[1]
                    pred_z = point[2]
                """

                 # exclude outlier points
                if(np.isnan(point[2]) or np.isinf(point[2])):
                    pred_x = 0.0
                    pred_y = 0.0
                    pred_z = 0.0
                    bad_n += 1

                save_file.write('{} {} {}\n'.format(pred_x, pred_y, pred_z))

        print(n)
        print("done!")
        print("Bad_n = {}".format(bad_n))

        save_file.close()


