# Week 5
Comparing Descriptor Matching Algorithms and Options

## Introduction
This week's blog will contain a comparison between BFMatcher and FLANN. It will also compare the results from changing
some of the parameters in BFMatcher. The descriptors used were generated from the AKAZE algorithm for feature detection
as it provided the best results.

## Brute-Force Matching (BFMatcher)
This method takes each feature from one of the descriptors and compares it to all the other features in the second one.
It returns the matching feature with minimal distance.

### BFMatcher with and without Cross Checking
The BFMatcher object takes two parameters: a `normType` and a boolean `crossCheck`. The first one is used to specify which type of distance measuring algorithm to use, such as: NORM_HAMMING, NORM_HAMMING2, NORM_L1, NORM_L2. The second parameter parameter is used to toggle Cross Checking. If it's true, then for any pair (i, j) found matching it will also check if it's true for (j, i). This means that enabling Cross Checking allows for a more consistent feature matching.

#### Code
```python
def bfmatcher(window, img1, kp1, desc1, img2, kp2, desc2, normType = cv2.NORM_HAMMING, crossCheck = True, noMatches = 250):
    bf = cv2.BFMatcher(normType = normType, crossCheck = crossCheck)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    matchesImage = np.zeros(shape=(1,1))
    matchesImage = cv2.drawMatches(img1 = img1, keypoints1 = kp1, img2 = img2, keypoints2 = kp2, matches1to2 = matches[:noMatches], outImg = matchesImage, flags=2)

    cv2.imwrite('BFMatcher.jpg', matchesImage)
    cv2.imshow('BFMatcher', matchesImage)
    cv2.waitKey(0)
```

### BFMatcher using the Ratio Test

A Ratio Test can also be used instead of Cross Cheking to check if the match is correct. This was proposed by D.Lowe[1] as a way of only getting correct matches for the SIFT algorithm, but it can also be used for AKAZE descriptors. For this, a modified function than the one above was used, which uses knnMatcher to get the best two matches of a feature. The match is only considered correct if the distance between the best two matches is more than 25%.

#### Code
```python
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

    cv2.imwrite('BFMatcherknn.jpg', matchesImage)
    cv2.imshow('BFMatcherknn', matchesImage)
    cv2.waitKey(0)
```

### Comparing several Distance Measuring Algorithms

#### NORM_HAMMING
Uses Hamming Distance to measure the distance between features.

| w/o Cross Checking | w/ Cross Checking | w/ Ratio Test |
| :---: | :---: | :---: |
| <img src="images/BFMatcher_NORM_HAMMING_without_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_HAMMING_with_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_HAMMING_with_RatioTest.jpg" width="300"> |

#### NORM_HAMMING2
Uses a modified Hamming Distance Algorithm to measure the distance between features. It's usually used for feature detection using ORB.

| w/o Cross Checking | w/ Cross Checking | w/ Ratio Test |
| :---: | :---: | :---: |
| <img src="images/BFMatcher_NORM_HAMMING2_without_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_HAMMING2_with_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_HAMMING2_with_RatioTest.jpg" width="300"> |

#### NORM_L1
Uses the Manhattan Distance to measure the distance between features.

| w/o Cross Checking | w/ Cross Checking | w/ Ratio Test |
| :---: | :---: | :---: |
| <img src="images/BFMatcher_NORM_L1_without_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_L1_with_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_L1_with_RatioTest.jpg" width="300"> |

#### NORM_L2
Uses Euclidian Distance to measure the distance between features.

| w/o Cross Checking | w/ Cross Checking | w/ Ratio Test |
| :---: | :---: | :---: |
| <img src="images/BFMatcher_NORM_L2_without_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_L2_with_CrossCheck.jpg" width="300"> | <img src="images/BFMatcher_NORM_L2_with_RatioTest.jpg" width="300"> |


All four algorithms for measuring distance provide similar results with NORM_HAMMING and NORM_L1 giving the best results. For binary based descriptors (such as AKAZE's descriptors), the OpenCV documentation recommends using NORM_HAMMING[2]. The images above also showed that Cross Checking provides better results, and the Ratio Test provides even better results in the best 250 matches.

### Comparing Cross Checking and Ratio Test on all the matches using NORM_HAMMING
| Cross Checking | Ratio Test |
| :---: | :---: |
| <img src="images/ALL_BFMatcher_NORM_HAMMING_with_CrossCheck.jpg" width="400"> | <img src="images/ALL_BFMatcher_NORM_HAMMING_with_RatioTest.jpg" width="400"> |

When all the feature matches are displayed, the Ratio Test only shows a small number of similarities where Cross Checking found a lot more. While the Ratio Test whould be better for finding objects from one image in another, this project tries to find the differences between the images, so the algorithm that finds the most similarities whould be a better candidate for the project.

## Fast Approximate Nearest Neighbor Search Matching (FLANN)
Another feature matching algorithm is FLANN. This uses nearest neighbours searching to find the best matches. It's a faster way of detecting matching features between two object than Brute-Force Matching. It was proposed by M.Muja and D.Lowe [3]. 

### Code
```python
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

    cv2.imwrite('FLANN.jpg', matchesImage)
    cv2.imshow('FLANN', matchesImage)
    cv2.waitKey(0)
```

### Comparing FLANN with BFMatcher
| FLANN | BFMatcher |
| :---: | :---: |
| <img src="images/FLANNMatcher.jpg" width="400"> | <img src="images/ALL_BFMatcher_NORM_HAMMING_with_CrossCheck.jpg" width="400"> |

FLANN provides similar results to the BFMatcher algorithm using the Ratio Test. While the matches are excellent, there are not enough matches to be able to then find the differences between the images.

## Conclusion
From all the test, the Brute-Force Matching algorithm using NORM_HAMMING and Cross Checking provided the best results for the next step of the project. This algorithm may have to be limited to only a subset of the best results as the last matches that BFMatcher provided seem to point completely in the wrong place.

## References
[1] D.Lowe, 'Distinctive Image Features from Scale-Invariant Keypoints', International Journal of Computer Vision, Vol. 60, 
Issue 2, 2004, pp. 91-110

[2] 'OpenCV 3.0-beta feature2d Tutorials', 2014, [Online]. Available: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html. [Accessed: 2017-10-14]

[3] M.Muja and D.Lowe, 'Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration', VISAPP International Conference on Computer Vision Theory and Applications, 2009
