# Task: Using the getMatches function, scale the second image to match the first one

import numpy as np
import cv2
import easygui
import math

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'fakePCB1.jpg')
pcb2 = cv2.imread(imagesPath + 'fakePCB2.jpg')

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
                'y': keyPoint1.pt[1],
                'angle': keyPoint1.angle
            },
            'pt2': {
                'x': keyPoint2.pt[0],
                'y': keyPoint2.pt[1],
                'angle': keyPoint2.angle
            }
        }

        matchedCoordinates.append(currentMatch)

    return matchedCoordinates

# matches = getMatches(pcb1, pcb2)

def getScalingLevel(matches):
    # get (X,Y) for each point in top 2 matches
    point1AX = matches[0]['pt1']['x']
    point1AY = matches[0]['pt1']['y']
    point2AX = matches[0]['pt2']['x']
    point2AY = matches[0]['pt2']['y']

    point1BX = matches[1]['pt1']['x']
    point1BY = matches[1]['pt1']['y']
    point2BX = matches[1]['pt2']['x']
    point2BY = matches[1]['pt2']['y']

    # get distance between the top two matches
    dist1 = getDistance(point1AX, point1AY, point1BX, point1BY)
    dist2 = getDistance(point2AX, point2AY, point2BX, point2BY)

    # return smaller distance divided by larger distance
    if(dist1 < dist2):
        return ((dist1 / dist2), 0)
    else:
        return ((dist2 / dist1), 1)


def scaleImage(img1, img2):
    matches = getMatches(img1, img2)
    scalingLevel, i = getScalingLevel(matches)

    # find image to scale
    if(i == 0):
        img = img2
    else:
        img = img1

    # get height and witdh of image
    h, w = np.shape(img)[:2]
    print(w, h)

    w = float(w)
    h = float(h)

    # resize to scalingLevel
    S = cv2.resize(img,(int(w*scalingLevel), int(h*scalingLevel)))

    if(i == 0):
        return (img1, S)
    else:
        return (S, img2)

pcb1, pcb2 = scaleImage(pcb1, pcb2);

h, w = np.shape(pcb1)[:2]
h1, w1 = np.shape(pcb2)[:2]
# # h2, w2 = np.shape(pcb2)[:2]
# # #cv2.imwrite(outputPath + 'scaledImage' + fileExtension, R)

# # print(w, h)


pcb1 = cv2.resize(pcb1, (int(w*0.25), int(h*0.25)))
pcb2 = cv2.resize(pcb2, (int(w1*0.25), int(h1*0.25)))
cv2.imwrite(outputPath + 'scaledImagePCB1' + fileExtension, pcb1)
cv2.imwrite(outputPath + 'scaledImagePCB2' + fileExtension, pcb2)
# # pcb2 = cv2.resize(pcb2,(int(w2*0.25), int(h2*0.25)))
# cv2.imshow('pcb1', pcb1)
# cv2.imshow('pcb2', pcb2)
# cv2.imshow('pcb2', pcb2)
cv2.waitKey(0)