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

        newImg1 = cv2.resize(src = img1, dsize = (int(width1 * scale), int(height1 * scale)), interpolation = cv2.INTER_AREA)
        newImg2 = cv2.resize(src = img2, dsize = (int(width2 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
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
    kp1, desc1 = akaze.detectAndCompute(image = img1, mask = None)
    kp2, desc2 = akaze.detectAndCompute(image = img2, mask = None)

    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(queryDescriptors = desc1, trainDescriptors = desc2)
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

    mask[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2 + offsetX)] = img[0:y2, 0:x2]

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

    return (rotationAngle)

def removeBorders(img):
    h, w = np.shape(img)[:2]

    B = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)

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

    M = cv2.getRotationMatrix2D(center = (cx,cy), angle = rotationAngle, scale = 1)
    R = cv2.warpAffine(src = img, M = M, dsize = (x, y), flags = cv2.INTER_CUBIC)

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

def matchRotation(img1, img2):
    borderedImg = addBorders(img2)
    rotationAngle = getRotationAngle(img1, img2)
    rotatedImage = rotateImage(borderedImg, rotationAngle)
    if(checkRotation(img1, rotatedImage) == False):
        rotatedImage = rotateImage(rotatedImage, 180)
    croppedImage = removeBorders(rotatedImage)

    return croppedImage


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
    S = cv2.resize(src = img, dsize = (int(w*scalingLevel), int(h*scalingLevel)), interpolation = cv2.INTER_CUBIC)

    if(i == 0):
        return (img1, S)
    else:
        return (S, img2)

def locationCorrection(img1, img2):
    (height, width) = img2.shape[:2]
    matches = getMatches(img1, img2)

    img1X = matches[0]['pt1']['x']
    img1Y = matches[0]['pt1']['y']
    img2X = matches[0]['pt2']['x']
    img2Y = matches[0]['pt2']['y']

    difX = img1X - img2X
    difY = img1Y - img2Y

    translationMatrix = np.float32([[1, 0, difX], [0, 1, difY]])
    transImg = cv2.warpAffine(src = img2, M = translationMatrix, dsize = (width, height), flags = cv2.INTER_LINEAR)

    return transImg

def getDifferences(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(image = img1, mask = None)
    kp2, desc2 = akaze.detectAndCompute(image = img2, mask = None)

    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(queryDescriptors = desc1, trainDescriptors = desc2)

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

def getMask(img1, img2):
    img1Height, img1Width = img1.shape[:2]

    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    (img1Dif, img2Dif) = getDifferences(img1, img2)

    mask = np.zeros((img1Height, img1Width, 1), np.uint8)
    mask[:, :] = 0
    for dif in img1Dif:
        mask[int(dif['y']), int(dif['x'])] = 255

    lastNoContours = len(img1Dif)

    shape = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))
    mask = cv2.dilate(src = mask, kernel = shape)

    for i in range(100):
        _, contours, _ = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(points = contour)

            patch = img1[y : y + h, x : x + w]
            (_, value) = getBestMatch(img2, patch)

            if value > 0.5:
                cv2.drawContours(mask, contour, -1, 0)
            else:
                cv2.drawContours(mask, contour, -1, 255, 3)

        noContours = len(contours)
        if noContours / lastNoContours < 0.1:
            lastNoContours = noContours
        else:
            break;

    return mask

def getBestMatch(img, patch):
    result = cv2.matchTemplate(image = img, templ = patch, method = cv2.TM_CCOEFF_NORMED)

    (_, value, _, (x, y)) = cv2.minMaxLoc(src = result)

    return ((x, y), value)

def getAllPatches(mask):
    patches = []

    _, contours, _ = cv2.findContours(image = mask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        arcPercentage = 0.01
        epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
        corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
        x, y, w, h = cv2.boundingRect(points = corners)
        currentArea = w * h

        # Ignore points
        if currentArea > 1:
            patches.append((x, y, w, h))

    return patches

def getBestPatches(sourceImg, checkImg, patches, threshold = 0.5):
    bestPatches = []
    for (x, y, w, h) in patches:
        patch = sourceImg[y : y + h, x : x + w]
        ((mX, mY), matchValue) = getBestMatch(checkImg, patch)
        if matchValue < threshold:
            bestPatches.append((x, y, w, h))

    return bestPatches

def getBestPatchesAuto(sourceImg, checkImg, patches):
    for th in range(100):
        threshold = th / 100.0
        bestPatches = getBestPatches(sourceImg, checkImg, patches, threshold)
        if len(bestPatches) > 0:
            return bestPatches
    return bestPatches

def addBar(img,new_width,new_height):
	newImg = np.zeros((new_height, new_width, 3), np.uint8)

	y1, x1 = np.shape(newImg)[:2]
	cx = x1/2
	cy = y1/2

	y2, x2 = np.shape(img)[:2]
	cx2 = x2/2
	cy2 = y2/2

    # Fix for odd sized images
	offsetX = x2 % 2
	offsetY = y2 % 2

	newImg[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2 + offsetX)] = img[0:y2, 0:x2]

	return newImg

def addBars(img1,img2):

	height1,width1=img1.shape[:2]
	height2,width2=img2.shape[:2]

	if width1 != width2:
		if width1 > width2:
			img2=addBar(img2,width1,height2)
			height2,width2=img2.shape[:2]

		else:
			img1=addBar(img1,width2,height1)
			height1,width1=img1.shape[:2]

	if height1 != height2:
		if height1>height2:

			img2=addBar(img2,width2,height1)

		else:

			img1=addBar(img1,width1,height2)

	return(img1,img2)


# TODO:
# Read Imgs with easygui
# ui using easygui
# Add intro comments

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'
file1 = imagesPath + 'pcb1.jpg'
file2 = imagesPath + 'pcb2.jpg'

img1 = cv2.imread(file1)
img2 = cv2.imread(file2)
(img1, img2) = downscaleImages(img1, img2)

img2 = matchRotation(img1, img2)
(img1, img2) = scaleImages(img1, img2)
img2 = locationCorrection(img1, img2)

gImg1 = cv2.cvtColor(src = img1, code = cv2.COLOR_BGR2GRAY)
gImg2 = cv2.cvtColor(src = img2, code = cv2.COLOR_BGR2GRAY)

mask = getMask(gImg1, gImg2)

patches = getAllPatches(mask)

(normImg1, normImg2) = normaliseImages(gImg1, gImg2)
bestPatches = getBestPatchesAuto(normImg1, normImg2, patches)

for (x, y, w, h) in bestPatches:
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 3)

(img1, img2) = addBars(img1, img2)

stackedImages = np.hstack((img1, img2))

cv2.imshow('Diffs', stackedImages)
cv2.waitKey(0)
