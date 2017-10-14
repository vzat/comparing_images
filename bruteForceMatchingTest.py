###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 14-10-2017
#
#	Title: Testing Brute Force Matching Algorithms

import numpy as np
import cv2
import easygui

imagesPath = 'images/'
outputPath = 'output/'
fileExtension = '.jpg'

def bfmatcher(window, img1, kp1, desc1, img2, kp2, desc2, normType = cv2.NORM_HAMMING, crossCheck = True, noMatches = 250):
    bf = cv2.BFMatcher(normType = normType, crossCheck = crossCheck)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    matchesImage = np.zeros(shape=(1,1))
    matchesImage = cv2.drawMatches(img1 = img1, keypoints1 = kp1, img2 = img2, keypoints2 = kp2, matches1to2 = matches[:noMatches], outImg = matchesImage, flags=2)

    cv2.imwrite(outputPath + window + fileExtension, matchesImage)
    cv2.imshow(window, matchesImage)
    cv2.waitKey(0)

def bfmatcherknn(window, img1, kp1, desc1, img2, kp2, desc2, normType = cv2.NORM_HAMMING, k = 2, noMatches = 250):
    bf = cv2.BFMatcher(normType = normType)
    matches = bf.knnMatch(desc1, desc2, k = 2)

    # Use D.Lowe's Ratio Test
    if k == 2:
        bestMatches = []
        for match1, match2 in matches:
            if match1.distance < 0.75 * match2.distance:
                bestMatches.append([match1])
        matches = bestMatches

    matchesImage = np.zeros(shape=(1,1))
    matchesImage = cv2.drawMatchesKnn(img1 = img1, keypoints1 = kp1, img2 = img2, keypoints2 = kp2, matches1to2 = matches[:noMatches], outImg = matchesImage, flags=2)

    cv2.imwrite(outputPath + window + fileExtension, matchesImage)
    cv2.imshow(window, matchesImage)
    cv2.waitKey(0)

def flannmatcher(window, img1, kp1, desc1, img2, kp2, desc2, flannIndex = 0, trees = 5, checks = 50, noMatches = 250):
    indexParams = dict(algorithm = flannIndex, trees = trees)
    searchParams = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    # Convert descriptors to CV_32F
    desc1 = np.float32(desc1)
    desc2 = np.float32(desc2)

    matches = flann.knnMatch(desc1, desc2, k = 2)

    # Use D.Lowe's Ratio Test
    mask = []
    matchIndex = 0
    for match1, match2 in matches:
        if matchIndex == noMatches:
            break
        matchIndex += 1
        if match1.distance < 0.75 * match2.distance:
            mask.append([1, 0])
        else:
            mask.append([0, 0])

    drawParams = dict(matchColor = (0, 0, 255), singlePointColor = (255, 0, 0), matchesMask = mask, flags = 0)

    matchesImage = np.zeros(shape=(1,1))
    matchesImage = cv2.drawMatchesKnn(img1 = img1, keypoints1 = kp1, img2 = img2, keypoints2 = kp2, matches1to2 = matches[:noMatches], outImg = matchesImage, **drawParams)

    cv2.imwrite(outputPath + window + fileExtension, matchesImage)
    cv2.imshow(window, matchesImage)
    cv2.waitKey(0)

pcb1 = cv2.imread(imagesPath + 'pcb1.jpg')
pcb2 = cv2.imread(imagesPath + 'pcb2.jpg')

gpcb1 = cv2.cvtColor(pcb1, cv2.COLOR_BGR2GRAY)
gpcb2 = cv2.cvtColor(pcb2, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()
kp1, desc1 = akaze.detectAndCompute(pcb1, None)
kp2, desc2 = akaze.detectAndCompute(pcb2, None)

# ### BFMatcher 250
# bfmatcher('BFMatcher_NORM_HAMMING_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING, True)
# bfmatcher('BFMatcher_NORM_HAMMING_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING, False)
# bfmatcher('BFMatcher_NORM_HAMMING2_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING2, True)
# bfmatcher('BFMatcher_NORM_HAMMING2_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING2, False)
# bfmatcher('BFMatcher_NORM_L1_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L1, True)
# bfmatcher('BFMatcher_NORM_L1_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L1, False)
# bfmatcher('BFMatcher_NORM_L2_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L2, True)
# bfmatcher('BFMatcher_NORM_L2_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L2, False)
#
# ### BFMatcher ALL
# bfmatcher('ALL_BFMatcher_NORM_HAMMING_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING, True, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_HAMMING_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING, False, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_HAMMING2_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING2, True, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_HAMMING2_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING2, False, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_L1_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L1, True, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_L1_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L1, False, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_L2_with_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L2, True, noMatches = None)
# bfmatcher('ALL_BFMatcher_NORM_L2_without_CrossCheck', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L2, False, noMatches = None)


# ### BFMatcher with knnMatch 250
# bfmatcherknn('BFMatcher_NORM_HAMMING_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING)
# bfmatcherknn('BFMatcher_NORM_HAMMING2_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING2)
# bfmatcherknn('BFMatcher_NORM_L1_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L1)
# bfmatcherknn('BFMatcher_NORM_L2_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L2)
#
# ### BFMatcher with knnMatch ALL
# bfmatcherknn('ALL_BFMatcher_NORM_HAMMING_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING, noMatches = None)
# bfmatcherknn('ALL_BFMatcher_NORM_HAMMING2_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_HAMMING2, noMatches = None)
# bfmatcherknn('ALL_BFMatcher_NORM_L1_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L1, noMatches = None)
# bfmatcherknn('ALL_BFMatcher_NORM_L2_with_RatioTest', pcb1, kp1, desc1, pcb2, kp2, desc2, cv2.NORM_L2, noMatches = None)


### FLANNMatcher ALL
flannmatcher('FLANNMatcher', pcb1, kp1, desc1, pcb2, kp2, desc2, checks = 50, noMatches = None)
