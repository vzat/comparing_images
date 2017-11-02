import numpy as np
import cv2
import easygui

def getBestMatch(img, patch):
    patchSize = patch.shape

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(image = gImg, templ = gPatch, method = cv2.TM_CCORR_NORMED)

    (_, value, _, (x, y)) = cv2.minMaxLoc(result)

    return ((x, y), value)

def getDistance(x1, y1, x2, y2):
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
        mask[dif['y'], dif['x']] = 255

    shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
    shape = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.erode(mask, shape, iterations = 1)
    shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, shape, iterations = 10)

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
        (_, matchValue)  = getBestMatch(checkImg, patch)
        # print matchValue
        # cv2.imshow('Patch', patch)
        # cv2.waitKey(0)
        if matchValue < threshold:
            bestPatches.append((x, y, w, h))

    return bestPatches

def eqImgs(img1, img2):
    (img1Height, img1Width) = img1.shape[:2]
    (img2Height, img2Width) = img2.shape[:2]

    yuv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
    (minI, maxI, _, _) = cv2.minMaxLoc(yuv1[:, :, 0])

    yuv1[:, :, 0] = 255 * ((yuv1[:, :, 0] - minI) / (maxI - minI))
    yuv2[:, :, 0] = 255 * ((yuv2[:, :, 0] - minI) / (maxI - minI))

    return (cv2.cvtColor(yuv1, cv2.COLOR_YUV2BGR), cv2.cvtColor(yuv2, cv2.COLOR_YUV2BGR))

def scaleImageDown(img):
    imgSize = np.shape(img)
    maxSize = (640, 480)
    if imgSize[0] > maxSize[0] or imgSize[1] > maxSize[1]:
        print 'Warning: Image too big'
        wRatio = float(float(maxSize[0]) / float(imgSize[0]))
        hRatio = float(float(maxSize[1]) / float(imgSize[1]))
        ratio = 1.0
        if wRatio > hRatio:
            ratio = wRatio
        else:
            ratio = hRatio
        return cv2.resize(img, (int(ratio * imgSize[1]), int(ratio * imgSize[0])))
    return img


imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

(pcb1, pcb2) = eqImgs(pcb1, pcb2)

# pcb1 = scaleImageDown(pcb1)
# pcb2 = scaleImageDown(pcb2)

# pcb1 = cv2.fastNlMeansDenoisingColored(pcb1,None,10,10,7,21)
# pcb2 = cv2.fastNlMeansDenoisingColored(pcb2,None,10,10,7,21)

mask = getMask(pcb1, pcb2)
patches = getAllPatches(mask)
bestPatches = getBestPatches(pcb1, pcb2, patches, 0.8)

for (x, y, w, h) in bestPatches:
    cv2.rectangle(pcb1, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('Differences', pcb1)
cv2.waitKey(0)
