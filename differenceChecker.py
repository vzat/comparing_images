###############################
#
#   (c) Vlad Zat, Emmet Doyle, Trent Nguyen, Cristian Anton 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 13-10-2017
#
#   Title - Difference Checker
#
#   Introduction:
#   - The difference checker is an application created for the purpose of difference that exists
#   in two images that are similar but have fundamental differences. This is achieved through the
#   equalization, image manipulation, morphological applications and template matching to the two
#   images allowing us to produce a final product that highlights the key differences between the one
#   of the images compared to the other.
#
#   Tested on OpenCV 3.2.0 with Python 2.7.
#
###############################

###############################
#
#   1. Background Information
#
#   Libraries:
#
#   CV2 - Extensive computer graphics library with sophisiticated algorithms and functions that can
#   be worked upon to suit the needs of many image/video application.
#
#   Numpy and Math - Mathematical libraries that are needed to sort through multidimensional array data
#   and perform equations to correct problems that arise in the application.
#
#   EasyGUI - A small graphifics librarie that allows for the quick and easy integration of UI to
#   applications.
#
###############################
#
#   2. Structure
#
#   2.1 Read in images
#   - Images are read in through here by manual navigation by the user. The images are stored in
#   image variables for use in the application.
#
#       * EasyGUI's fileopenbox() function was used to allow the user to sift through the files on
#       their PC allowing them to navigate to the image files. Once the images are loaded the button
#       can be pressed and will start the application.
#
#   2.2 Image Manipulation and feature detection
#   - The images are passed through several functions that downscale, upscale, rotate and transform them
#   so that they are better suited for the operations to detect the differences. During this process the
#   features are detected and stored in an array so that they can be used later in the application as a
#   basis of finding the differences.
#
#       2.2.1 Downscale image
#       - Big images often cause trouble for applications as they increase run time and worse produce an
#       abundant amount of false positives which would otherwise not show up if we downscale the image.
#       Here we downscale our image so that we can display our results properly, reduce runtime and noise.
#
#           * In our downscaleImages() we extract the width and height of the images to find out which one of
#           the images is bigger. By finding the bigger image we can compared it to the smaller image thus
#           creating a scaling factor which we'll use to ensure that the two images are the small dimensions.
#           The resizing was done with the cv2.resize() function.
#
#       2.2.2 Rotating the image with feature detection
#       - We will try to rotate the second image to match the orientation of the object in the first image. This
#       allows our algorithms to perform better as the objects to be detected will be put as close together as they
#       can be on the x and y planes.
#
#           * In matchRotation() we first add a border to our second image (i.e. the one to be rotated). This is because
#           when we rotate the image we will lose data due to the slight shift which we want to keep until the rotation is
#           complete. Next, we put the images through our getMatches() function to get the features that exist in the both
#           images. This function uses AKAZE as a feature detector [1] .Using the features we pass the two best lines to
#           our getRotationAngle() function and by using the x and y coordinates we calcalate the atan of the two lines
#           and apply this angle to the second image. We then rotate the second image using this angle and return the rotated image.
#
#       2.2.4 Location Correction
#       - The location correction will move the objects within the second image as close as possible to the object in the first using
#       the features. As the rotation of the image had been corrected we will simply find the sum needed to be applied to the axis to
#        bring the second object to roughly the same position as the first.
#
#           * The coordinates are loaded in the locationCorrection() function. The getMatches() will be used to once more find the differences
#             in the images and using by using the difference we find the sum needed to be transitioned. A translation matrix is created with
#             the x and y axis difference [2]. The warpAffine() function will take this in and apply it to our image.
#
#   2.3 Getting the mask
#   - The mask of the differences will be populated here. Using histogram equalization techniques, morphological operations and contour
#   sizing comparisons we are able to create a mask that display areas that are dense where differences exists.
#
#       2.3.1 Creating the mask
#       - The initial mask is created with the differences and will be dilated to ensure that positives will cojoin together and be shown as a
#       an area of difference. At the same time, false positive which are often isolated will be removed thus creating a mask of areas of difference.
#
#           * In our getMask() function we first use equalizeHist() so that the distribution of pixels is even across the image. By use of our
#             getDifferences() function we populate a newly created image with features that are not present in either images (the differences)
#             as getDifferences() stores the differences in an array. This function use AKAZE for feature detection [1].
#             Template matching is then applied to every contour seeing if it doesn't exist in the second image.
#             If it did we were able to automate our "dilation" loop which works by drawing a black contour on pixels less than a certain size thus
#             coloring them black while drawing a white contour around pixels that have conjoined.
#             We repeat this until there are no more iterations that can be made. Just using dilation would makes all the pixels bigger.
#             This only increases the biggest differences.
#
#   2.4 Applying Template Matching
#   - Using the mask attained from 2.3 we are able to draw a contour around all the differences discovered. We apply CLAHE to the images to sharpen the image
#   and by using template matching we can search to see if patches exist on the second image compared to the first. Thus we were able to isolate only the
#   true differences in the two images.
#
#       2.4.1 Getting the patches
#       - In our getAllPatches() we apply our cv2.boundingRect() to the image where patches exist thus creating an image where every patch,
#       even the noise, is apparent.
#
#       2.4.2 Applying CLAHE
#       - We now apply Contrasted Limited Adaptive Histogram Equalation to the images thus allowing the distribution to be sharp through the image.
#           * In our normaliseImages() function we apply the CLAHE to the entire image rather than just segments of the image. The contrast
#           clipping was set to 40 and the reason for us apply it to the whole image is because we wanted the image to equalize in relation towil
#           itself rather than have sharp points throughout.
#
#       2.4.3 Getting the best patches
#       - The patches that appear in both and aren't similar to each other are retrieved in this section. This is where only the true difference
#       is retrieved from the application and is drawn to the image.
#           * getBestPatches() takes in a list of contours and a theshold. It then uses template matching to find the normalised matched value.
#           It then returns all the contours which are smalled than the threshold. The values for the normalised template matching can be between
#           0.0 and 1.0 where 1.0 is a perfect match [3]
#           * Our getBestPatchAuto() will try multiple thesholds and determine the lowest theshold where patches are found. This makes sure that
#           only the best patches are returned.
#
#   2.5 Displaying the images
#   - The contour is applied to this image and then displayed using hstack next to its compared image.
#
###############################

###############################
#
#   3. Extra Notes:
#
#       * While template matching is better at detecting differences, it would take too long to check every pixel in the images.
#         This is why the features are filtered using feature detection first.
#
#       * Two types of normalisation are used in the project. While CLAHE normalises the images better, it creates a lot of noise.
#         Noise is very detrimental to the feature detection algorithm. This is why equalize hist is used for feature detection.
#         When the false-positive contours are eliminated using template matching, CLAHE is used as it makes the images match even more.
#
###############################

###############################
#
#   4. References
#
#   [1] 'AKAZE local features matching', 2014, [Online].
#       Available: http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/akaze_matching/akaze_matching.html.
#       [Accessed: 2017-10-05]
#
#   [2] R.Szeliski, 'Feature-based alignment' in 'Computer Vision: Algorithms and Applications', 2010, Springer, p. 312
#
#   [3] U.Sinha, 'Template matching', 2010, [Online].
#       Available: http://www.aishack.in/tutorials/template-matching.
#       [Accessed: 2017-10-26]
#
###############################

import numpy as np
import cv2
import easygui
from math import atan

def downscaleImages(img1, img2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    maxWidth = 1000.0
    if width1 > maxWidth or width2 > maxWidth:
        if width1 > maxWidth and width1 > width2:
            scale = maxWidth / width1
        else:
            scale = maxWidth / width2

        newImg1 = cv2.resize(src = img1, dsize = (int(width1 * scale), int(height1 * scale)), interpolation = cv2.INTER_AREA)
        newImg2 = cv2.resize(src = img2, dsize = (int(width2 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
    else:
        newImg1 = img1.copy()
        newImg2 = img2.copy()

    return (newImg1, newImg2)

def normaliseImages(img1, img2):
    height1, width1 = np.shape(img1)[:2]
    height2, width2 = np.shape(img2)[:2]

    if height1 * width1 > height2 * width2:
        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (width1, height1))
    else:
        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (width2, height2))

    return (clahe.apply(img1), clahe.apply(img2))

def getMatches(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(image = img1, mask = None)
    kp2, desc2 = akaze.detectAndCompute(image = img2, mask = None)

    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(queryDescriptors = desc1, trainDescriptors = desc2)
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

def getDiameter(img):
    h, w = np.shape(img)[:2]
    hyp = (w*w + h*h)**(1/2.0)
    return (int(hyp)+1)

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

    mask[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2 + offsetX)] = img[0:y2, 0:x2]

    return (mask)

def getRotationAngle(img1, img2):
    matches = getMatches(img1, img2)

    point1AX = matches[0]['pt1']['x']
    point1AY = matches[0]['pt1']['y']
    point2AX = matches[0]['pt2']['x']
    point2AY = matches[0]['pt2']['y']

    point1BX = matches[1]['pt1']['x']
    point1BY = matches[1]['pt1']['y']
    point2BX = matches[1]['pt2']['x']
    point2BY = matches[1]['pt2']['y']

    m1 = ((point1BY - point1AY) / (point1BX - point1AX))
    line1Angle = atan(m1)

    m2 = ((point2BY - point2AY) / (point2BX - point2AX))
    line2Angle = atan(m2)

    rotationAngle = (line2Angle - line1Angle)

    rotationAngle = np.rad2deg(rotationAngle)

    return (rotationAngle)

def removeBorders(img):
    h, w = np.shape(img)[:2]

    B = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)

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

    M = cv2.getRotationMatrix2D(center = (cx,cy), angle = rotationAngle, scale = 1)
    R = cv2.warpAffine(src = img, M = M, dsize = (x, y), flags = cv2.INTER_CUBIC)

    return R

def checkRotation(img1, img2):
    matches = getMatches(img1, img2)

    point1AX = matches[0]['pt1']['x']
    point2AX = matches[0]['pt2']['x']

    point1BX = matches[1]['pt1']['x']
    point2BX = matches[1]['pt2']['x']

    if(point1AX < point1BX and point2AX > point2BX):
        return False
    elif(point1AX > point1BX and point2AX < point2BX):
        return False
    else:
        return True

def matchRotation(img1, img2):
    print 'Rotating Images'
    borderedImg = addBorders(img2)
    rotationAngle = getRotationAngle(img1, img2)
    rotatedImage = rotateImage(borderedImg, rotationAngle)
    if(checkRotation(img1, rotatedImage) == False):
        rotatedImage = rotateImage(rotatedImage, 180)
    croppedImage = removeBorders(rotatedImage)

    return croppedImage


def getDistance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2.0)

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

def scaleImages(img1, img2):
    print 'Scaling Images'
    matches = getMatches(img1, img2)
    scalingLevel, i = getScalingLevel(matches)

    # find image to scale
    if(i == 0):
        img = img2
    else:
        img = img1

    # get height and witdh of image
    h, w = np.shape(img)[:2]

    w = float(w)
    h = float(h)

    # resize to scalingLevel
    S = cv2.resize(src = img, dsize = (int(w*scalingLevel), int(h*scalingLevel)), interpolation = cv2.INTER_CUBIC)

    if(i == 0):
        return (img1, S)
    else:
        return (S, img2)

def locationCorrection(img1, img2):
    print 'Translating Images'

    (height, width) = img2.shape[:2]
    matches = getMatches(img1, img2)

    img1X = matches[0]['pt1']['x']
    img1Y = matches[0]['pt1']['y']
    img2X = matches[0]['pt2']['x']
    img2Y = matches[0]['pt2']['y']

    difX = img1X - img2X
    difY = img1Y - img2Y

    translationMatrix = np.float32([[1, 0, difX], [0, 1, difY]])
    transImg = cv2.warpAffine(src = img2, M = translationMatrix, dsize = (width, height), flags = cv2.INTER_LINEAR)

    return transImg

def getDifferences(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(image = img1, mask = None)
    kp2, desc2 = akaze.detectAndCompute(image = img2, mask = None)

    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(queryDescriptors = desc1, trainDescriptors = desc2)

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
    print 'Searching for differences'
    img1Height, img1Width = img1.shape[:2]

    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    (img1Dif, img2Dif) = getDifferences(img1, img2)

    mask = np.zeros((img1Height, img1Width, 1), np.uint8)
    mask[:, :] = 0
    for dif in img1Dif:
        mask[int(dif['y']), int(dif['x'])] = 255

    lastNoContours = len(img1Dif)

    shape = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))
    mask = cv2.dilate(src = mask, kernel = shape)

    for i in range(100):
        _, contours, _ = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(points = contour)

            patch = img1[y : y + h, x : x + w]
            (_, value) = getBestMatch(img2, patch)

            if value > 0.5:
                cv2.drawContours(mask, contour, -1, 0)
            else:
                cv2.drawContours(mask, contour, -1, 255, 3)

        noContours = len(contours)
        if noContours / lastNoContours < 0.1:
            lastNoContours = noContours
        else:
            break;

    return mask

def getBestMatch(img, patch):
    result = cv2.matchTemplate(image = img, templ = patch, method = cv2.TM_CCOEFF_NORMED)

    (_, value, _, (x, y)) = cv2.minMaxLoc(src = result)

    return ((x, y), value)

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
    print 'Eliminating false-positives'
    for th in range(100):
        threshold = th / 100.0
        bestPatches = getBestPatches(sourceImg, checkImg, patches, threshold)
        if len(bestPatches) > 0:
            return bestPatches
    return bestPatches

def addBar(img,new_width,new_height):
	newImg = np.zeros((new_height, new_width, 3), np.uint8)

	y1, x1 = np.shape(newImg)[:2]
	cx = x1/2
	cy = y1/2

	y2, x2 = np.shape(img)[:2]
	cx2 = x2/2
	cy2 = y2/2

    # Fix for odd sized images
	offsetX = x2 % 2
	offsetY = y2 % 2

	newImg[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2 + offsetX)] = img[0:y2, 0:x2]

	return newImg

def addBars(img1,img2):

	height1,width1=img1.shape[:2]
	height2,width2=img2.shape[:2]

	if width1 != width2:
		if width1 > width2:
			img2=addBar(img2,width1,height2)
			height2,width2=img2.shape[:2]

		else:
			img1=addBar(img1,width2,height1)
			height1,width1=img1.shape[:2]

	if height1 != height2:
		if height1>height2:

			img2=addBar(img2,width2,height1)

		else:

			img1=addBar(img1,width1,height2)

	return(img1,img2)

applicationSwitch = True
file1Ver = False
file2Ver = False

while applicationSwitch:
    title = 'Difference Checker'
    instruction = 'Please load image 1 and 2 then begin.'

    if file1Ver == False or file2Ver == False:
        buttons = ['Load Image 1', 'Load Image 2']
    else:
        buttons = ['Load Image 1', 'Load Image 2', 'Begin Application']

    selection = easygui.indexbox(msg = instruction, title = title, choices = buttons)

    if selection == 0:
        file1 = easygui.fileopenbox()
        img1 = cv2.imread(file1)
        if img1 is None:
            easygui.msgbox("Please select image files only!")
        else:
            file1Ver = True
    elif selection == 1:
        file2 = easygui.fileopenbox()
        img2 = cv2.imread(file2)
        if img2 is None:
            easygui.msgbox("Please select image files only!")
        else:
            file2Ver = True
    elif selection == 2:
        (img1, img2) = downscaleImages(img1, img2)

        img2 = matchRotation(img1, img2)
        (img1, img2) = scaleImages(img1, img2)
        img2 = locationCorrection(img1, img2)

        gImg1 = cv2.cvtColor(src = img1, code = cv2.COLOR_BGR2GRAY)
        gImg2 = cv2.cvtColor(src = img2, code = cv2.COLOR_BGR2GRAY)

        mask = getMask(gImg1, gImg2)

        patches = getAllPatches(mask)

        (normImg1, normImg2) = normaliseImages(gImg1, gImg2)
        bestPatches = getBestPatchesAuto(normImg1, normImg2, patches)

        for (x, y, w, h) in bestPatches:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 3)

        (img1, img2) = addBars(img1, img2)

        stackedImages = np.hstack((img1, img2))

        print 'Done'
        cv2.imshow('Differences', stackedImages)
        cv2.waitKey(0)
        applicationSwitch = False
    else:
        applicationSwitch = False
