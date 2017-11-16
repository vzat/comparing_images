import numpy as np
import cv2
import easygui

def getBestMatch(img, patch):
    patchSize = patch.shape

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # cv2.TM_CCOEFF_NORMED or cv2.TM_CCORR_NORMED
    result = cv2.matchTemplate(image = gImg, templ = gPatch, method = cv2.TM_CCOEFF_NORMED)

    (_, value, _, (x, y)) = cv2.minMaxLoc(result)

    return ((x, y), value)


imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

(tHeight, tWidth) = (5, 5)

height, width = pcb1.shape[:2]

mask = np.zeros((height, width, 1), np.uint8)
for x in range(tWidth / 2, width - tWidth / 2, 5):
    for y in range(tHeight / 2, height - tHeight / 2, 5):
        patch = pcb1[y - tHeight / 2 : y + tHeight / 2, x - tWidth / 2 : x + tWidth / 2]
        (_, value) = getBestMatch(pcb2, patch)

        if value < 0.5:
            mask[y][x] = 255

        print x, y

cv2.imshow('Mask', mask)
cv2.waitKey(0)
