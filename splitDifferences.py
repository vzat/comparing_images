# Task: using morphology, modify the mask to combine the pixels as much as possible
# but at the same time it should contain multiple components
# E.g. mask1 contains multiple components but has a lot of noise
# while mask2 is clean but only has one part

import numpy as np
import cv2
import easygui

imagesPath = 'images/'

mask = cv2.imread(imagesPath + 'mask.jpg')

shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)

cv2.imshow('Mask1', mask1)

shape = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 10))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)
shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 30))
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, shape)

cv2.imshow('Mask2', mask2)
cv2.waitKey(0)
