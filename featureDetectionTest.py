###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 04-10-2017
#
#	Title: Testing Feature Detection Algorithms

import numpy as np
import cv2
import easygui

# help(cv2.drawKeypoints)
# help(cv2.drawMatches)

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

I1 = cv2.imread(imagesPath + 'pcb1.jpg')
I2 = cv2.imread(imagesPath + 'pcb2.jpg')

G1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
G2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

def displayFAST(window, image, nms=1):
    # FAST - Features from Accelerated Segment Test
    # Reference - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
    fast = cv2.FastFeatureDetector_create()

    # Set maximum suppression - nms ignores unimportant pixels in corners that are not the the local maxima
    # Reference - http://users.ecs.soton.ac.uk/msn/book/new_demo/nonmax/
    fast.setNonmaxSuppression(nms)

    # find corners
    keyPoints = fast.detect(image, None)
    corners = image.copy()
    corners = cv2.drawKeypoints(image = image, keypoints = keyPoints, outImage = corners, color = (0, 0, 255))

    cv2.imwrite(outputPath + window + fileExtension, corners)
    cv2.imshow(window, corners)

def displayORB(window, image):
    # ORB - Oriented FAST and Rotated BRIEF
    # Reference - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb
    orb = cv2.ORB_create()

    keyPoints = orb.detect(image, None)
    keyPoints, descriptor = orb.compute(image, keyPoints)
    newImage = image.copy()
    newImage = cv2.drawKeypoints(image = image, keypoints = keyPoints, outImage = newImage, color = (0, 0, 255), flags = 0)

    cv2.imwrite(outputPath + window + fileExtension, newImage)
    cv2.imshow(window, newImage)

def displayAKAZE(window, image):
    # AKAZE
    # Reference - http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/akaze_matching/akaze_matching.html
    akaze = cv2.AKAZE_create()

    keyPoints = akaze.detect(image, None)
    newImage = image.copy()
    newImage = cv2.drawKeypoints(image = image, keypoints = keyPoints, outImage = newImage, color = (0, 0, 255), flags = 0)

    cv2.imwrite(outputPath + window + fileExtension, newImage)
    cv2.imshow(window, newImage)

def showDifs(window, image1, keyPoints1, desc1, image2, keyPoints2, desc2):
    # Feature Matching
    # Reference - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    matchesImage = image1.copy()
    matchesImage = cv2.drawMatches(img1 = image1, keypoints1 = keyPoints1, img2 = image2, keypoints2 = keyPoints2, matches1to2 = matches[:250], outImg = matchesImage, flags=2)

    cv2.imwrite(outputPath + window + fileExtension, matchesImage)
    cv2.imshow(window, matchesImage)

# displayFAST('fast1', I1)
# displayFAST('fast2', I2)
#
# displayORB('orb1', I1)
# displayORB('orb2', I2)
#
# displayAKAZE('akaze1', I1)
# displayAKAZE('akaze2', I2)

# Get edges from images with Canny Edge Detection - TODO: test with multiple feature detection algorithms
# Reference http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
# edges1 = cv2.Canny(I1, 150, 200)
# edges2 = cv2.Canny(I2, 150, 200)
# cv2.imshow('canny1', edges1)
# cv2.imshow('canny2', edges2)

# Noise Reduction using 5x5 Gaussian filter ?

# ORB
# orb = cv2.ORB_create()
# keyPoints1, desc1 = orb.detectAndCompute(I1, None)
# keyPoints2, desc2 = orb.detectAndCompute(I2, None)

# AKAZE
akaze = cv2.AKAZE_create()
keyPoints1, desc1 = akaze.detectAndCompute(I1, None)
keyPoints2, desc2 = akaze.detectAndCompute(I2, None)

showDifs('akazeDif', I1, keyPoints1, desc1, I2, keyPoints2, desc2)

key = cv2.waitKey(0)
