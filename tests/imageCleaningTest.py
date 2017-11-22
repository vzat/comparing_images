import cv2
import numpy as np
from matplotlib import pyplot as plt

imagesPath = 'images/'
outputPath = 'output/filters/'
fileExtension = '.jpg'

def convolution2D(img, window):
   
  kernel = np.ones((5,5),np.float32)/25
  dst = cv2.filter2D(img,-1,kernel)


  imgs = np.hstack((img, dst))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)


def averaging(img, window):

  blur = cv2.blur(img,(5,5))
   
  imgs = np.hstack((img, blur))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)

def gaussianBlur(img, window):
  blur = cv2.GaussianBlur(img,(5,5),0)

  imgs = np.hstack((img, blur))

  cv2.imshow(window, imgs)
  cv2.imwrite(outputPath + window + fileExtension, imgs)

def medianBlur(img, window):

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
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

#convolution2D(pcb1, "convolution1")
#convolution2D(pcb2, "convolution2")
#averaging(pcb1, "averaging1")
#averaging(pcb2, "averaging2")
#gaussianBlur(pcb1, "gaussianBlur1")
#gaussianBlur(pcb2, "gaussianBlur2")
#medianBlur(pcb1, "medianBlur")
#bilateralFilter(pcb1, "bilateralFilter1")
#bilateralFilter(pcb2, "bilateralFilter2")
fastN1MeansDenoising(pcb1, "Denoised1")
fastN1MeansDenoising(pcb2, "Denoised2")

key = cv2.waitKey(0)