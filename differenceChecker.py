import numpy as np
import cv2
from math import atan

def downscaleImages(img1, img2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    maxWidth = 1000.0
    if width1 > maxWidth or width2 > maxWidth:
        if width1 > maxWidth and width1 > width2:
            scale = maxWidth / width1
        else:
            scale = maxWidth / width2

        newImg1 = cv2.resize(img1, (int(width1 * scale), int(height1 * scale)), interpolation = cv2.INTER_AREA)
        newImg2 = cv2.resize(img2, (int(width2 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
    else:
        newImg1 = img1.copy()
        newImg2 = img2.copy()

    return (newImg1, newImg2)

def normaliseImages(img1, img2):
    height1, width1 = np.shape(img1)[:2]
    height2, width2 = np.shape(img2)[:2]

    if height1 * width1 > height2 * width2:
        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (width1, height1))
    else:
        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (width2, height2))

    return (clahe.apply(img1), clahe.apply(img2))

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

def getDiameter(img):
    h, w = np.shape(img)[:2]
    hyp = (w*w + h*h)**(1/2.0)
    return (int(hyp)+1)

def addBorders(img):
    hyp = getDiameter(img)
    mask = np.zeros((hyp, hyp, 3), np.uint8)

    y1, x1 = np.shape(mask)[:2]
    cx = x1/2
    cy = y1/2

    y2, x2 = np.shape(img)[:2]
    cx2 = x2/2
    cy2 = y2/2

    # Fix for odd sized images
    offsetX = x2 % 2
    offsetY = y2 % 2

    mask[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2+offsetY)] = img[0:y2, 0:x2]

    return (mask)

def getRotationAngle(img1, img2):
    matches = getMatches(img1, img2)

    point1AX = matches[0]['pt1']['x']
    point1AY = matches[0]['pt1']['y']
    point2AX = matches[0]['pt2']['x']
    point2AY = matches[0]['pt2']['y']

    point1BX = matches[1]['pt1']['x']
    point1BY = matches[1]['pt1']['y']
    point2BX = matches[1]['pt2']['x']
    point2BY = matches[1]['pt2']['y']

    m1 = ((point1BY - point1AY) / (point1BX - point1AX))
    line1Angle = atan(m1)

    m2 = ((point2BY - point2AY) / (point2BX - point2AX))
    line2Angle = atan(m2)

    rotationAngle = (line2Angle - line1Angle)

    rotationAngle = np.rad2deg(rotationAngle)

    # angleDif = []
    # for match in matches:
    #     angle1 = match['pt1']['angle']
    #     angle2 = match['pt2']['angle']
    #     angleDif.append(angle2 - angle1)
    #
    # angleDif = sorted(angleDif)
    # rotationAngle = np.average(angleDif)
    # print rotationAngle

    return (rotationAngle)

def removeBorders(img):
    h, w = np.shape(img)[:2]

    B = img.copy()
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

def rotateImage(img, rotationAngle):
    y, x = np.shape(img)[:2]
    cx = x/2
    cy = y/2

    M = cv2.getRotationMatrix2D((cx,cy), rotationAngle, 1)
    # cv2.INTER_NEAREST, cv2.INTER_LINEAR (default), cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
    R = cv2.warpAffine(img, M, (x, y))

    return R

def checkRotation(img1, img2):
    matches = getMatches(img1, img2)

    point1AX = matches[0]['pt1']['x']
    point2AX = matches[0]['pt2']['x']

    point1BX = matches[1]['pt1']['x']
    point2BX = matches[1]['pt2']['x']

    if(point1AX < point1BX and point2AX > point2BX):
        return False
    elif(point1AX > point1BX and point2AX < point2BX):
        return False
    else:
        return True


def getDistance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2.0)

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

def scaleImages(img1, img2):
    matches = getMatches(img1, img2)
    scalingLevel, i = getScalingLevel(matches)

    # find image to scale
    if(i == 0):
        img = img2
    else:
        img = img1

    # get height and witdh of image
    h, w = np.shape(img)[:2]

    w = float(w)
    h = float(h)

    # resize to scalingLevel
    S = cv2.resize(img,(int(w*scalingLevel), int(h*scalingLevel)))

    if(i == 0):
        return (img1, S)
    else:
        return (S, img2)

def matchRotation(img1, img2):
    borderedImg = addBorders(img2)
    rotationAngle = getRotationAngle(img1, img2)
    rotatedImage = rotateImage(borderedImg, rotationAngle)
    if(checkRotation(img1, rotatedImage) == False):
        rotatedImage = rotateImage(rotatedImage, 180)
    rotatedImage2 = removeBorders(rotatedImage2)

# TODO: Replace with easygui
imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'
file1 = imagesPath + 'pcb1.jpg'
file2 = imagesPath + 'pcb2.jpg'

img1 = cv2.imread(file1)
img2 = cv2.imread(file2)
(img1, img2) = downscaleImages(img1, img2)


normImg1 = img1.copy()
normImg2 = img2.copy()


borderedImg2 = addBorders(normImg2)
rotationAngle = getRotationAngle(normImg1, normImg2)
rotatedImage2 = rotateImage(borderedImg2, rotationAngle)
if(checkRotation(normImg1, rotatedImage2) == False):
    rotatedImage2 = rotateImage(rotatedImage2, 180)
rotatedImage2 = removeBorders(rotatedImage2)

(normImg1, normImg2) = scaleImages(normImg1, normImg2)

gImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
(normImg1, normImg2) = normaliseImages(gImg1, gImg2)


cv2.imshow('1', normImg1)
cv2.imshow('2', rotatedImage2)
cv2.waitKey(0)
