import numpy as np
import cv2
import easygui

def findBestMatch(window, img, patch, method, maxLoc = True):
    patchSize = patch.shape

    result = cv2.matchTemplate(image = img, templ = patch, method = method)

    if maxLoc:
        (_, r, _, (x, y)) = cv2.minMaxLoc(result)
    else:
        (r, _, (x, y), _) = cv2.minMaxLoc(result)

    print window, r

    # cv2.imshow('Result', result)
    # cv2.waitKey(0)

    cv2.rectangle(img, (x, y), (x + patchSize[1], y + patchSize[0]), (255, 0, 0), 3)

    cv2.imshow(window, img)
    cv2.imwrite(outputPath + window + fileExtension, img)
    cv2.waitKey(0)

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

img1Size = pcb1.shape
img2Size = pcb2.shape

# patch = pcb1[300:400, 350:450]
# patch = pcb1[180:280, 200:475]
# patch = pcb1[162:162+218, 236:236+105]
patch = pcb1[162:162+105, 236:236+218]
cv2.imshow('Patch', patch)
cv2.waitKey(0)
patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
# threshold, _ = cv2.threshold(src = patch, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# patch = cv2.Canny(image = patch.copy(), threshold1 = 0.5 * threshold, threshold2 = threshold)

patchSize = patch.shape

# https://docs.opencv.org/trunk/d4/dc6/tutorial_py_template_matching.html
methods = [ 'cv2.TM_CCOEFF',
            'cv2.TM_CCOEFF_NORMED',
            'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED',
            'cv2.TM_SQDIFF',
            'cv2.TM_SQDIFF_NORMED']

for methodName in methods:
    method = eval(methodName)
    # img = pcb1.copy()
    # GRAYSCALE WORKS BETTER ???
    # OR just edges (Canny) ???
    img = cv2.cvtColor(pcb2, cv2.COLOR_BGR2GRAY)
    # threshold, _ = cv2.threshold(src = img, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img = cv2.Canny(image = img.copy(), threshold1 = 0.5 * threshold, threshold2 = threshold)

    if methodName == 'cv2.TM_SQDIFF' or methodName == 'cv2.TM_SQDIFF_NORMED':
        findBestMatch(methodName, img, patch, method, False)
    else:
        findBestMatch(methodName, img, patch, method)

# while True:
#     for methodName in methods:
#         img = pcb1.copy()
#         method = eval(methodName)
#
#         result = cv2.matchTemplate(image = pcb1.copy(), templ = patch.copy(), method = method)
#         (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
#
#         if methodName == 'cv2.TM_SQDIFF' or methodName == 'cv2.TM_SQDIFF_NORMED':
#             cv2.rectangle(img, minLoc, (minLoc[0] + patchSize[0], minLoc[1] + patchSize[1]), (255, 0, 0), 3)
#         else:
#             cv2.rectangle(img, maxLoc, (maxLoc[0] + patchSize[0], maxLoc[1] + patchSize[1]), (255, 0, 0), 3)
#         cv2.imshow(methodName, img)
