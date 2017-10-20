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

# pcb1 = cv2.fastNlMeansDenoisingColored(pcb1, None, 10, 10, 7, 21)
# pcb2 = cv2.fastNlMeansDenoisingColored(pcb2, None, 10, 10, 7, 21)
# pcb1 = cv2.bilateralFilter(pcb1, 9, 75, 75)
# pcb2 = cv2.bilateralFilter(pcb2, 9, 75, 75)

img1Size = pcb1.shape
img2Size = pcb2.shape

def getDistance(x1, y1, x2, y2):
    # return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2.0)

def getMatches(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda match:match.distance)

    matchedCoordinates = []
    for match in matches:
        keyPoint1 = kp1[match.queryIdx]
        keyPoint2 = kp2[match.trainIdx]

        currentMatch = {
            'pt1': {
                'x': keyPoint1.pt[0],
                'y': keyPoint1.pt[1]
            },
            'pt2': {
                'x': keyPoint2.pt[0],
                'y': keyPoint2.pt[1]
            },
            'distance': match.distance
        }

        matchedCoordinates.append(currentMatch)

    return matchedCoordinates

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
    mask[dif['y'], dif['x']] = 255

cv2.imwrite(outputPath + 'mask1' + fileExtension, mask)

shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)

cv2.imwrite(outputPath + 'mask2' + fileExtension, mask)

_, contours, _ = cv2.findContours(image = mask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
bestContour = contours[0]
maxArea = 0
for contour in contours:
    #       [1] OpenCV, 'Contour Approximation', 2015. [Online].
    #           Available: http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    #           [Accessed: 2017-10-20]
    arcPercentage = 0.01
    epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
    corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
    x, y, w, h = cv2.boundingRect(points = corners)
    currentArea = w * h

    if currentArea > maxArea:
        maxArea = currentArea
        bestContour = corners

    # Ignore points
    if currentArea > 1:
        cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)

# x, y, w, h = cv2.boundingRect(points = bestContour)
# cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)



# cv2.imshow('mask', mask)
cv2.imshow('pcb1', pcb1)
cv2.imwrite(outputPath + 'diffs' + fileExtension, pcb1)
# cv2.imshow('pcb2', pcb2)
cv2.waitKey(0)



# Creating a mask with white pixels
# mask = np.zeros((img1Size[0], img1Size[1], 1), np.uint8)
# mask[:, :] = 0
# for match in matches:
#     pt2 = match['pt2']
#     mask[int(pt2['y']), int(pt2['x'])] = 255


# Useful for combining pixels
# shape = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
# shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
# shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 10))
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
# shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 30))
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
# mask = cv2.bitwise_not(mask)

# _, contours, h = cv2.findContours(image = mask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(pcb2, contours, -1, (0, 0, 255), 3)



# WORKS ???
# TODO Invert colours so the differences are detected instead of the same things
# imgArea = img2Size[0] * img2Size[1]
# maxArea = 0
# bestContour = contours[0]
# for contour in contours:
#     arcPercentage = 0.01
#     epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
#     corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
#     x, y, w, h = cv2.boundingRect(points = corners)
#     currentArea = w * h
#
#     if currentArea < imgArea and maxArea < currentArea:
#         maxArea = currentArea
#         bestContour = corners
#
#     cv2.rectangle(pcb2, (x, y), (x + w, y + h), (0, 0, 255), 3)





# cv2.drawContours(pcb2, bestContour, -1, (0, 0, 255), 3)
# x, y, w, h = cv2.boundingRect(points = bestContour)
# cv2.rectangle(pcb2, (x, y), (x + w, y + h), (0, 0, 255), 3)



# newImg = pcb2[y : y + h, x : x + w]
# newMask = mask[y : y + h, x : x + w]
#
# _, contours, h = cv2.findContours(image = newMask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
# for contour in contours:
#     arcPercentage = 0.01
#     epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
#     corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
#     x, y, w, h = cv2.boundingRect(points = corners)
#     cv2.rectangle(newImg, (x, y), (x + w, y + h), (0, 0, 255), 3)
#
# cv2.imshow('mask', newMask)
# cv2.imshow('img', newImg)

# _, contours, hierarchy = cv2.findContours(image = mask.copy(), mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE)
# for i, contour in enumerate(contours):
#     if hierarchy[0][i][0] > 0:
#         cv2.drawContours(pcb2, contour, -1, (0, 0, 255), 3)

# cv2.imshow('mask', mask)
# # cv2.imshow('pcb1', pcb1)
# cv2.imshow('pcb2', pcb2)
# cv2.waitKey(0)
