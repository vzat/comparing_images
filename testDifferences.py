###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 14-10-2017
#
#	Title: Testing methods for finding differences between images

import numpy as np
import cv2
import easygui

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

def getMatches(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)
    print kp1[0]

    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    return bf.match(desc1, desc2)

matches = getMatches(pcb1, pcb2)
#
# for match in matches:
#     # Dist between keypoints
#     print match.distance
#
#     # Img1 keypoint
#     print match.queryIdx
#
#     # Img2 keypoint
#     print match.trainIdx
#
#     # Train kp index
#     print match.imgIdx
#
    # print ''
