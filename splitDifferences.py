# Task: using morphology, modify the mask to combine the pixels as much as possible
# but at the same time it should contain multiple components
# E.g. mask1 contains multiple components but has a lot of noise
# while mask2 is clean but only has one part

import numpy as np
import cv2
import easygui

from matplotlib import pyplot as plt
from matplotlib import image as image

def getDifferences(img1, img2):
    # Detect and Compute keypoints and descriptors for each image
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

    # Get a list of all matches
    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc1, desc2)

    # Mark keypoints that were matched previously
    for match in matches:
        kp1[match.queryIdx] = None
        kp2[match.trainIdx] = None

    # Create a list that contains the coordinate of the keypoints for the first image
    differences1 = []
    for keyPoint in kp1:
        if keyPoint != None:
            currentKP = {
                'x': keyPoint.pt[0],
                'y': keyPoint.pt[1]
            }
            differences1.append(currentKP)
            
    # Create a list that contains the coordinate of the keypoints for the second image
    differences2 = []
    for keyPoint in kp2:
        if keyPoint != None:
            currentKP = {
                'x': keyPoint.pt[0],
                'y': keyPoint.pt[1]
            }
            differences2.append(currentKP)

    return (differences1, differences2)

imagesPath = 'images/'

mask = cv2.imread(imagesPath + 'mask.jpg')

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

shape1Dim = (10, 10)
shape1DimSmoother = (3, 3)

shape1 = cv2.getStructuringElement(cv2.MORPH_RECT, shape1Dim)
mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape1)

#Change the shape to get a finer structure
shape1 = cv2.getStructuringElement(cv2.MORPH_RECT, shape1DimSmoother)

#Removes the noise and keeps the components
mask1 = cv2.erode(mask1, shape1, iterations = 1)

#Enhances the components
mask1 = cv2.dilate(mask1, shape1, iterations = 10)

shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

cv2.imshow('Mask1', mask1)

shape = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 10))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 30))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)

mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

img1Size = pcb1.shape

(img1Dif, img2Dif) = getDifferences(pcb1, mask1)

mask3 = np.zeros((img1Size[0], img1Size[1], 1), np.uint8)
mask3[:, :] = 0
for dif in img1Dif:
    mask[dif['y'], dif['x']] = 255

img1Size = pcb1.shape

(img1Dif, img2Dif) = getDifferences(pcb1, pcb2)

_, contours, _ = cv2.findContours(image = mask3.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
for contour in contours:
    arcPercentage = 0.01
    epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
    corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
    x, y, w, h = cv2.boundingRect(points = corners)
    currentArea = w * h

    # Ignore points
    if currentArea > 1:
        cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('Mask1', mask1)
cv2.imshow('Mask2', mask2)
cv2.imshow('pcb1', pcb1)

cv2.waitKey(0)
cv2.destroyAllWindows()
