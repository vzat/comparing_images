# Week 5
Comparing Descriptor Matching Algorithms and Options

## Introduction
This week's blog will contain a comparison between BFMatcher and FLANN. It will also compare the results from changing
some of the parameters in BFMatcher. The descriptors used were generated from the AKAZE algorithm for feature detection
as it provided the best results.

## Brute-Force Matching
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



## Fast Approximate Nearest Neighbor Search Matching (FLANN)

## References
[1] D.Lowe, 'Distinctive Image Features from Scale-Invariant Keypoints', International Journal of Computer Vision, Vol. 60, 
Issue 2, 2004, pp. 91-110

