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
print img1Size, img2Size

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
for match in matches:
    pt1 = match['pt1']
    pt2 = match['pt2']
    radius = 5
    cv2.circle(pcb1, (int(pt1['x']), int(pt1['y'])), radius, (0, 0, 255), -1)
    cv2.circle(pcb2, (int(pt2['x']), int(pt2['y'])), radius, (0, 0, 255), -1)
    # pcb1[pt1['y'], pt1['x']] = (0, 0, 255)
    # pcb2[pt2['y'], pt2['x']] = (0, 0, 255)

cv2.imshow('pcb1', pcb1)
cv2.imshow('pcb2', pcb2)
cv2.waitKey(0)
