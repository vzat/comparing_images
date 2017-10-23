import numpy as np
import cv2
import easygui

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

    result = cv2.matchTemplate(image = pcb1.copy(), templ = patch.copy(), method = method)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    print minLoc, maxLoc

cv2.imshow('Patch', patch)
cv2.waitKey(0)
