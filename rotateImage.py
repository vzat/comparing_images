# Task: Using the getMatches function, rotate and move the second image to match the first one

import numpy as np
import cv2
import easygui

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

# Calculate Euclidean Distance between 2 points
def getDistance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2.0)

# Returns a sorted list with the matched pixels
# THe list has the following format:
# matches
#   -> pt1 (coordinates from the first image)
#       -> x
#       -> y
#   -> pt2 (coordinates from the second image)
#       -> x
#       -> y
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
            }
        }

        matchedCoordinates.append(currentMatch)

    return matchedCoordinates

matches = getMatches(pcb1, pcb2)

# Use the matches list to find the orientation of the second image
# and rotate it to match the first one
# You can access the values like this: matches['pt1']['x']
# to get the x coordinate from the first image

cv2.imshow('pcb1', pcb1)
cv2.imshow('pcb2', pcb2)
cv2.waitKey(0)
