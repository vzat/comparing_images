# Week 6
Finding unmatched features

## Introduction
The research from the past two weeks have shown that the best way to find the common features between two images is to use AKAZE as the Feature Detection algorithm and Brute-Force Matching with NORM_HAMMING and Cross Checking enabled. This week's blog will show the first steps in finding the differences between images.

## Step 1: Extracting the different features

For this step, a list will be generated for each image containing only the keypoints that were not matched by the Brute-Force Matching algorithm. This will result in potential features that could be different. This part works by extracting the keypoints and descriptors for each image. Using Brute-Force Matching, the matches from the descriptors. After this a list is created for each image that contains the coordinates of each feature that wasn't matched.

### Code
```python
def getDifferences(img1, img2):
    # Detect and Compute keypoints and descriptors for each image
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(img1, None)
    kp2, desc2 = akaze.detectAndCompute(img2, None)

    # Get a list of all matches
    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(desc1, desc2)

    # Mark keypoints that were matched previously
    for match in matches:
        kp1[match.queryIdx] = None
        kp2[match.trainIdx] = None

    # Create a list that contains the coordinate of the keypoints for the first image
    differences1 = []
    for keyPoint in kp1:
        if keyPoint != None:
            currentKP = {
                'x': keyPoint.pt[0],
                'y': keyPoint.pt[1]
            }
            differences1.append(currentKP)
            
    # Create a list that contains the coordinate of the keypoints for the second image
    differences2 = []
    for keyPoint in kp2:
        if keyPoint != None:
            currentKP = {
                'x': keyPoint.pt[0],
                'y': keyPoint.pt[1]
            }
            differences2.append(currentKP)

    return (differences1, differences2)
```
