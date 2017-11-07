# Task: Using the getMatches function, rotate and move the second image to match the first one

import numpy as np
import cv2
import easygui
import math

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

matches = getMatches(pcb1, pcb2)


# Use the matches list to find the orientation of the second image
# and rotate it to match the first one
# You can access the values like this: matches['pt1']['x']
# to get the x coordinate from the first image


def getRotationAngle(matches):
    point1AX = matches[0]['pt1']['x']
    point1AY = matches[0]['pt1']['y']
    point2AX = matches[0]['pt2']['x']
    point2AY = matches[0]['pt2']['y']

    point1BX = matches[1]['pt1']['x']
    point1BY = matches[1]['pt1']['y']
    point2BX = matches[1]['pt2']['x']
    point2BY = matches[1]['pt2']['y']

    angle1 = matches[0]['pt1']['angle']
    angle2 = matches[0]['pt2']['angle']

    # m1 = ((point1BY - point1AY) / (point1BX - point1AX))
    # line1Angle = math.atan(m1)
    #
    # m2 = ((point2BY - point2AY) / (point2BX - point2AX))
    # line2Angle = math.atan(m2)
    #
    # rotationAngle = line2Angle - line1Angle
    #
    # rotationAngle = np.rad2deg(rotationAngle)

    return angle2 - angle1


rotationAngle = getRotationAngle(matches)

print(rotationAngle)


def getDiameter(img):
    h, w = np.shape(img)[:2]
    hyp = (w*w + h*h)**(1/2.0)
    return(int(hyp)+1)

def addBorders(img):
    hyp = getDiameter(img)
    mask = np.zeros((hyp, hyp, 3), np.uint8)

    y1, x1 = np.shape(mask)[:2]
    cx = x1/2
    cy = y1/2

    y2, x2 = np.shape(img)[:2]
    cx2 = x2/2
    cy2 = y2/2

    mask[int(cy-cy2):int(cy+cy2) , int(cx-cx2):int(cx+cx2)] = img[0:y2, 0:x2]

    return(mask)

borderedImg = addBorders(pcb2)

def removeBorders(img):
    h, w = np.shape(img)[:2]

    B = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left = w
    right = 1
    top = h
    bottom = 1

    for i in range (1, h):
        for j in range (1, w):
            if B[i,j] > 0:

                if i < top:
                    top = i

                if i > bottom:
                    bottom = i

                if j < left:
                    left = j

                if j > right:
                    right = j

    C = img[top:bottom, left:right]

    return C



def rotateImage(img):
    y, x = np.shape(img)[:2]
    cx = x/2
    cy = y/2

    M = cv2.getRotationMatrix2D((cx,cy), rotationAngle, 1)
    R = cv2.warpAffine(img, M, (x, y))

    return R

R = rotateImage(borderedImg)

def scaleImage(img1, img2):
    # get shape of two images
    # scale second image to the first
    y1, x1 = np.shape(img1)[:2]
    y2, x2 = np.shape(img2)[:2]

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    if x1 >= x2:
        scaleW = x2/x1
        w = x1
    else:
        scaleW = x1/x2
        w = x2

    if y1 >= y2:
        scaleH = y2/y1
        h = y1
    else:
        scaleH = y1/y2
        h = y2

    S = cv2.resize(img2,(int(w*scaleW), int(h*scaleH)))

    return S

cropped = removeBorders(R)
Result = scaleImage(pcb1, cropped)

cv2.imshow('Result', Result)
# cv2.imshow('pcb1', pcb1)
cv2.imshow('cropped', cropped)
cv2.imwrite(outputPath + 'addBorders' + fileExtension, borderedImg)
# cv2.imwrite(outputPath + 'R' + fileExtension, R)
cv2.waitKey(0)
