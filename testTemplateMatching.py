import numpy as np
import cv2
import easygui

def findBestMatch(window, img, patch, method, maxLoc = True):
    patchSize = patch.shape

    result = cv2.matchTemplate(image = img, templ = patch, method = method)

    if maxLoc:
        (_, _, _, (x, y)) = cv2.minMaxLoc(result)
    else:
        (_, _, (x, y), _) = cv2.minMaxLoc(result)

    cv2.rectangle(img, (x, y), (x + patchSize[0], y + patchSize[1]), (255, 0, 0), 3)

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

patch = pcb1[300:400, 350:450]
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
    img = pcb1.copy()

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
