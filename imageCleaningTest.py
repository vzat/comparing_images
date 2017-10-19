import cv2
import numpy as np
from matplotlib import pyplot as plt

imagesPath = 'images/'
outputPath = 'output/filters/'
fileExtension = '.jpg'

def convolution2D(img):
   
  kernel = np.ones((5,5),np.float32)/25
  dst = cv2.filter2D(img,-1,kernel)


  imgs = np.hstack((img, dst))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)


def averaging(img):

  blur = cv2.blur(img,(5,5))
   
  imgs = np.hstack((img, blur))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)

def gaussianBlur(img):
  blur = cv2.GaussianBlur(img,(5,5),0)

  imgs = np.hstack((img, blur))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)

def medianBlur(img):

  median = cv2.medianBlur(img,5)

  imgs = np.hstack((img, median))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)

def bilateralFilter(img, window):

  blur = cv2.bilateralFilter(img,9,75,75)

  imgs = np.hstack((img, blur))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)
  # key = cv2.waitKey(0)

def fastN1MeansDenoising(img, window):

  dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

  imgs = np.hstack((img, dst))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')

#convolution2D(pcb1, "convolution")
#averaging(pcb1, "averaging")
#gaussianBlur(pcb1, "gaussianBlur")
#medianBlur(pcb1, "medianBlur")
bilateralFilter(pcb1, "bilateralFilter")
fastN1MeansDenoising(pcb1, "Denoised")

key = cv2.waitKey(0)