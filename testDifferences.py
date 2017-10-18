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

matches = getMatches(pcb1, pcb2)

# minDist = img1Size[0]
# noMatches = len(matches)
# for index1 in range(0, noMatches - 1):
#     for index2 in range(index1 + 1, noMatches):
#         coord1 = matches[index1]['pt1']
#         coord2 = matches[index2]['pt1']
#         currentDistance = getDistance(coord1['x'], coord1['y'], coord2['x'], coord2['y'])
#         if currentDistance < minDist:
#             minDist = currentDistance
# print minDist


# for match in matches:
#     pt1 = match['pt1']
#     pt2 = match['pt2']
#     radius = 5
#     cv2.circle(pcb1, (int(pt1['x']), int(pt1['y'])), radius, (0, 0, 255), -1)
#     cv2.circle(pcb2, (int(pt2['x']), int(pt2['y'])), radius, (0, 0, 255), -1)

mask = np.zeros((img1Size[0], img1Size[1], 1), np.uint8)
mask[:, :] = 0
for match in matches:
    pt2 = match['pt2']
    mask[int(pt2['y']), int(pt2['x'])] = 255
    # radius = 1
    # cv2.circle(mask, (int(pt1['x']), int(pt1['y'])), radius, (255, 255, 255), -1)

shape = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
# mask = cv2.bitwise_not(mask)

_, contours, _ = cv2.findContours(image = mask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(pcb2, contours, -1, (0, 0, 255), 3)

# TODO Invert colours so the differences are detected instead of the same things
for contour in contours:
    arcPercentage = 0.1
    epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
    corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
    x, y, w, h = cv2.boundingRect(points = corners)
    currentArea = w * h

    cv2.rectangle(pcb2, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('mask', mask)
# cv2.imshow('pcb1', pcb1)
cv2.imshow('pcb2', pcb2)
cv2.waitKey(0)
