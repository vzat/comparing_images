# Task: using morphology, modify the mask to combine the pixels as much as possible
# but at the same time it should contain multiple components
# E.g. mask1 contains multiple components but has a lot of noise
# while mask2 is clean but only has one part

import numpy as np
import cv2
import easygui

from matplotlib import pyplot as plt
from matplotlib import image as image

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

img1Size = pcb1.shape
img2Size = pcb2.shape

def getDifferences(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc1, desc2)

    for match in matches:
        kp1[match.queryIdx] = None
        kp2[match.trainIdx] = None

    differences1 = []
    differences2 = []

    for keyPoint in kp1:
        if keyPoint != None:
            currentKP = {
                'x': keyPoint.pt[0],
                'y': keyPoint.pt[1]
            }
            differences1.append(currentKP)

    for keyPoint in kp2:
        if keyPoint != None:
            currentKP = {
                'x': keyPoint.pt[0],
                'y': keyPoint.pt[1]
            }
            differences2.append(currentKP)

    return (differences1, differences2)

# matches = getMatches(pcb1, pcb2)
(img1Dif, img2Dif) = getDifferences(pcb1, pcb2)

mask = np.zeros((img1Size[0], img1Size[1], 1), np.uint8)

mask[:, :] = 0

for dif in img1Dif:
    mask[int(dif['y']), int(dif['x'])] = 255

mask = cv2.resize(mask, dsize = (int(0.5*img1Size[1]), int(0.5*img1Size[0])))

shape1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape1)

#Change the shape to get a finer structure
shape1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

#Removes the noise and keeps the components
mask1 = cv2.erode(mask1, shape1, iterations = 2)

# Increase the kernel to make the rois larger
shape1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#Enhances the components
mask1 = cv2.dilate(mask1, shape1, iterations = 4)

#cv2.imshow('Mask1', mask1)

cv2.imshow("Mask", mask1)

pcb1 = cv2.resize(pcb1, dsize = (int(0.5*img1Size[1]), int(0.5*img1Size[0])))

_, contours, _ = cv2.findContours(image = mask1.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
for contour in contours:
    arcPercentage = 0.01
    epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
    corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
    x, y, w, h = cv2.boundingRect(points = corners)
    currentArea = w * h

    # Ignore points
    if currentArea > 1:
        cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('pcb1', pcb1)

# shape = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
# mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
# shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
# mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
# shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 10))
# mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
# shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 30))
# mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
#
# cv2.imshow('Mask2', mask2)

cv2.waitKey(0)
cv2.destroyAllWindows()