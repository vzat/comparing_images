import numpy as np
import cv2

def normaliseImg(img):
    height, width = np.shape(img)[:2]
    clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (width, height))

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    luminance = yuv[:, :, 0]
    yuv[:, :, 0] = clahe.apply(luminance)

    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def getBestMatch(img, patch):
    patchSize = patch.shape

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gImg = img.copy()
    gPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # gPatch = patch.copy()

    # cv2.TM_CCOEFF_NORMED or cv2.TM_CCORR_NORMED
    result = cv2.matchTemplate(image = gImg, templ = gPatch, method = cv2.TM_CCOEFF_NORMED)

    (_, value, _, (x, y)) = cv2.minMaxLoc(result)

    return ((x, y), value)

def getDistance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2.0)

def getDifferences(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

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

def getMask(img1, img2):
    img1Height, img1Width = pcb1.shape[:2]

    (img1Dif, img2Dif) = getDifferences(img1, img2)

    mask = np.zeros((img1Height, img1Width, 1), np.uint8)
    mask[:, :] = 0
    for dif in img1Dif:
        mask[int(dif['y']), int(dif['x'])] = 255

    # maxTh = 50
    # _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # initContours = len(contours)
    # for th in range(maxTh):
    #     newMask = mask.copy()
    #     for dif1 in img1Dif:
    #         for dif2 in img1Dif:
    #             if getDistance(dif1['x'], dif1['y'], dif2['x'], dif2['y']) < th:
    #                 cv2.line(newMask, (int(dif1['x']), int(dif1['y'])), (int(dif2['x']), int(dif2['y'])), 255)
    #
    #     _, contours, _ = cv2.findContours(newMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     print len(contours)
    #     cv2.imshow('NewMask', newMask)
    #     cv2.waitKey(0)

    # mask = newMask.copy()

    # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # noComp = len(contours)
    # for i in range(100):
    #     mask = cv2.dilate(mask, shape)
    #     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     curComp = len(contours)
    #     if curComp < noComp * 50 / 100:
    #         break
    #
    # noComp = curComp
    # for i in range(100):
    #     mask = cv2.erode(mask, shape)
    #     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     curComp = len(contours)
    #     print curComp
    #     if curComp > noComp + noComp * 10 / 100:
    #         break

    shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, shape, iterations = 10)
    shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, shape, iterations = 22)

    cv2.imshow('Mask', mask)
    cv2.waitKey(0)


    # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # noComp = len(contours)
    # for i in range(100):
    #     mask = cv2.dilate(mask, shape)
    #     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     curComp = len(contours)
    #     if curComp < noComp * 50 / 100:
    #         break
    #
    # noComp = curComp
    # for i in range(100):
    #     mask = cv2.erode(mask, shape)
    #     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     curComp = len(contours)
    #     print curComp
    #     if curComp > noComp * 105 / 100:
    #         break


    # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
    # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # mask = cv2.erode(mask, shape, iterations = 1)
    # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask = cv2.dilate(mask, shape, iterations = 10)

    return mask

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
    for th in range(10):
        threshold = th / 10.0
        bestPatches = getBestPatches(normPCB1, normPCB2, patches, threshold)
        if len(bestPatches) > 0:
            return bestPatches
    return bestPatches

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

def getRotationAngle(matches):
    # avgAngle = 0
    #
    # if len(matches) < 5:
    #     noMatches = len(matches)
    # else:
    #     noMatches = 5
    #
    # for matchNo in range(noMatches):
    #     angle1 = matches[matchNo]['pt1']['angle']
    #     angle2 = matches[matchNo]['pt2']['angle']
    #
    #     avgAngle += (angle2 - angle1)
    #
    # avgAngle /= noMatches

    # for match in matches:
    #     angle1 = match['pt1']['angle']
    #     angle2 = match['pt2']['angle']
    #
    #     print angle2 - angle1


    angle1 = matches[0]['pt1']['angle']
    angle2 = matches[0]['pt2']['angle']

    return angle2 - angle1

    # return avgAngle

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

    # Fix for odd sized images
    offsetX = x2 % 2
    offsetY = y2 % 2

    mask[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2+offsetY)] = img[0:y2, 0:x2]

    return(mask)

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

def rotateImage(img, rotationAngle):
    y, x = np.shape(img)[:2]
    cx = x/2
    cy = y/2

    M = cv2.getRotationMatrix2D((cx,cy), rotationAngle, 1)
    R = cv2.warpAffine(img, M, (x, y))

    return R

# def scaleImage(img1, img2):
#     # get shape of two images
#     # scale second image to the first
#     y1, x1 = np.shape(img1)[:2]
#     y2, x2 = np.shape(img2)[:2]
#
#     x1 = float(x1)
#     y1 = float(y1)
#     x2 = float(x2)
#     y2 = float(y2)
#
#     if x1 >= x2:
#         scaleW = x2/x1
#         w = x1
#     else:
#         scaleW = x1/x2
#         w = x2
#
#     if y1 >= y2:
#         scaleH = y2/y1
#         h = y1
#     else:
#         scaleH = y1/y2
#         h = y2
#
#     S = cv2.resize(img2,(int(w*scaleW), int(h*scaleH)))
#
#     return S

def locationCorrection(img1, img2):
    height, width = np.shape(img2)[:2]

    gImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gImg1 = cv2.medianBlur(gImg1, 9)
    # gImg2 = cv2.medianBlur(gImg2, 9)
    #
    # threshold, _ = cv2.threshold(src = gImg1, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # gImg1 = cv2.Canny(image = gImg1, threshold1 = 0.5 * threshold, threshold2 = threshold)
    #
    # threshold, _ = cv2.threshold(src = gImg2, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # gImg2 = cv2.Canny(image = gImg2, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # Convert frames to CV_32FC1
    gImg1_32 = np.float32(gImg1)
    gImg2_32 = np.float32(gImg2)

    # Find the translation between the two frames
    (xDif, yDif), _ = cv2.phaseCorrelate(src1 = gImg1_32, src2 = gImg2_32)

    translationMatrix = np.float32([[1, 0, xDif], [0, 1, yDif]])
    return cv2.warpAffine(img2, translationMatrix, (width, height))

def transImgs(img1, img2):
    (height, width) = img2.shape[:2]
    matches = getMatches(img1, img2)

    img1X = matches[0]['pt1']['x']
    img1Y = matches[0]['pt1']['y']
    img2X = matches[0]['pt2']['x']
    img2Y = matches[0]['pt2']['y']

    difX = img1X - img2X
    difY = img1Y - img2Y

    translationMatrix = np.float32([[1, 0, difX], [0, 1, difY]])
    transImg = cv2.warpAffine(img2, translationMatrix, (width, height))

    return (img1, transImg)

    # cv2.imshow('Translated', transImg)
    # cv2.waitKey(0)

def cleanPatch(img2, patch):
    pHeight, pWidth = patch.shape[:2]
    oPatch = patch.copy()

    (sHeight, sWidth) = (25, 25)
    if pHeight > sHeight and pWidth > sWidth:
        sX = sY = 0
        while (sY + sHeight < pHeight):
            sPatch = patch[sY : sY + sHeight, sX : sX + sWidth]
            (_, value) = getBestMatch(img2, sPatch)

            if value < 0.25:
                cv2.rectangle(oPatch, (sX, sY), (sX + sWidth, sY + sHeight), (0, 0, 0), 1)

            sX += sWidth
            if sX + sWidth > pWidth:
                sX = 0
                sY += sHeight

            print value

    cv2.imwrite(outputPath + 'cleanPatch' + fileExtension, oPatch)
    # cv2.imshow('Original Patch', oPatch)
    # cv2.waitKey(0)

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

def scaleImagesDown(img1, img2):
    # Scale the images down if they are too big

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    maxWidth = 1000.0
    if width1 > maxWidth or width2 > maxWidth:
        if width1 > maxWidth and width1 > width2:
            scale = maxWidth / width1
        else:
            scale = maxWidth / width2

        newImg1 = cv2.resize(img1, (int(width1 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
        newImg2 = cv2.resize(img2, (int(width2 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
    else:
        newImg1 = img1.copy()
        newImg2 = img2.copy()

    return (newImg1, newImg2)

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

(pcb1, pcb2) = scaleImagesDown(pcb1, pcb2)

# height1, width1 = pcb1.shape[:2]
# height2, width2 = pcb2.shape[:2]
#
# pcb1 = cv2.resize(pcb1, (int(width1 * 0.25), int(height1 * 0.25)))
# pcb2 = cv2.resize(pcb2, (int(width2 * 0.25), int(height2 * 0.25)))

normPCB1 = normaliseImg(pcb1)
normPCB2 = normaliseImg(pcb2)

# normPCB1 = cv2.GaussianBlur(normPCB1, (5, 5), 0)
# normPCB2 = cv2.GaussianBlur(normPCB2, (5, 5), 0)


# normPCB1 = cv2.fastNlMeansDenoisingColored(normPCB1, None, 3, 3, 7, 21)
# normPCB2 = cv2.fastNlMeansDenoisingColored(normPCB2, None, 3, 3, 7, 21)
#
# normPCB1 = cv2.medianBlur(normPCB1, 3)
# normPCB2 = cv2.medianBlur(normPCB2, 3)

# cv2.imshow('1', normPCB1)
# cv2.imshow('2', normPCB2)


# cv2.imwrite(outputPath + 'normPCB1' + fileExtension, normPCB1)
# cv2.imwrite(outputPath + 'normPCB2' + fileExtension, normPCB2)



# matches = getMatches(normPCB1, normPCB2)
# rotationAngle = getRotationAngle(matches)
# borderedImg = addBorders(normPCB2)
# R = rotateImage(borderedImg, rotationAngle)
# cropped = removeBorders(R)
# normPCB2 = cropped.copy()
# normPCB2 = scaleImage(normPCB1, cropped)

(normPCB1, normPCB2) = scaleImage(normPCB1, normPCB2)

(normPCB1, normPCB2) = transImgs(normPCB1, normPCB2)

# cv2.imshow('dsaad', normPCB1)
# cv2.imshow('dsaadaas', normPCB2)
# cv2.waitKey(0)

mask = getMask(normPCB1, normPCB2)
patches = getAllPatches(mask)

# bestPatches = getBestPatches(normPCB1, normPCB2, patches, 0.9)
bestPatches = getBestPatchesAuto(normPCB1, normPCB2, patches)

# pX, pY, pW, pH = bestPatches[0]
# patch = normPCB1[pY : pY + pH, pX : pX + pW]
# cleanPatch(normPCB2, patch)

for (x, y, w, h) in bestPatches:
    cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('Differences', pcb1)
# cv2.imwrite(outputPath + 'newDifferences' + fileExtension, pcb1)

# cv2.imshow('PCB1', normPCB1)
# cv2.imshow('PCB2', normPCB2)
cv2.waitKey(0)
