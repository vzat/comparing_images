import numpy as np
import cv2

def normaliseImg(img):
    height, width = np.shape(img)[:2]
    clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (height, width))

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
    mask = cv2.erode(mask, shape, iterations = 20)

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

    mask[int(cy-cy2):int(cy+cy2) , int(cx-cx2):int(cx+cx2)] = img[0:y2, 0:x2]

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


imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

normPCB1 = normaliseImg(pcb1)
normPCB2 = normaliseImg(pcb2)

cv2.imwrite(outputPath + 'normPCB1' + fileExtension, normPCB1)
cv2.imwrite(outputPath + 'normPCB2' + fileExtension, normPCB2)



matches = getMatches(normPCB1, normPCB2)
rotationAngle = getRotationAngle(matches)
borderedImg = addBorders(normPCB2)
R = rotateImage(borderedImg, rotationAngle)
cropped = removeBorders(R)
normPCB2 = scaleImage(normPCB1, cropped)


mask = getMask(normPCB1, normPCB2)
patches = getAllPatches(mask)





# bestPatches = getBestPatches(normPCB1, normPCB2, patches, 0.9)
bestPatches = getBestPatchesAuto(normPCB1, normPCB2, patches)

for (x, y, w, h) in bestPatches:
    cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('Differences', pcb1)
cv2.imwrite(outputPath + 'newDifferences' + fileExtension, pcb1)

# cv2.imshow('PCB1', normPCB1)
# cv2.imshow('PCB2', normPCB2)
cv2.waitKey(0)
