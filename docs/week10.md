# Week 10
Bug fixing and improvements

## Introduction
This blog post covers the improvements, experiments and bug fixes implemented
in the last week. These enhancements and issues were discovered from testing
the solution on other images by the team members.

## Bug Fixes
One major bug found was causing the program to crash on images with an odd width
or height. The problem was narrowed down to the `addBorders()` function, which
adds a border to an image before rotating it. It was caused by the inaccuracy of
casting a float to int when calculating the centre of an image. To fix the odd
number of pixels problem, an offset was added to the values if the number of pixels
was odd.

```python
    y1, x1 = np.shape(mask)[:2]
    cx = x1/2
    cy = y1/2

    y2, x2 = np.shape(img)[:2]
    cx2 = x2/2
    cy2 = y2/2

    # Fix for odd sized images
    offsetX = x2 % 2
    offsetY = y2 % 2

    mask[int(cy-cy2):int(cy+cy2 + offsetY) , int(cx-cx2):int(cx+cx2+offsetY)] = img[0:y2, 0:x2]
```

Another bug found by testing was the incorrect order of the parameters when creating
an CLAHE object. More specifically, the order of the width and height was wrong.
The CLAHE object required the `tileGridSize` parameter to be a tuple containing
the width and height of the grid size. The original code had the values in the tuple
the wrong way around.

```python
clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (width, height))
```

## Improvements
In the progress of testing the solution of other images, some improvements were
implemented. These improvements help the program work better on some images.

### Image Translation
For the images to be mapped to each other, one of the images has to be moved.
To find the offset between them, the best match from the feature detection
algorithm was used. By subtracting the x and y coordinates of the same key point,
the offset was found. After this, a translation matrix was formed using these values [1].
A translation matrix has the following form:

| 1 0 tx |

| 0 1 ty |

After the matrix was constructed, the `warpAffine()` function from OpenCV was used
to apply it to one of the images.

```python
def mapImgs(img1, img2):
    (height, width) = img2.shape[:2]
    matches = getMatches(img1, img2)

    img1X = matches[0]['pt1']['x']
    img1Y = matches[0]['pt1']['y']
    img2X = matches[0]['pt2']['x']
    img2Y = matches[0]['pt2']['y']

    difX = img1X - img2X
    difY = img1Y - img2Y

    translationMatrix = np.float32([[1, 0, difX], [0, 1, difY]])
    mappedImg = cv2.warpAffine(img2, translationMatrix, (width, height))

    return (img1, mappedImg)
```

### Downscaling Images
Another problem was discovered while testing other images. Large images can
not only slow down the program, but can also impede the process of finding differences.
To fix this problem, large images are downscaled before any processing is done on them.

```python
def scaleImagesDown(img1, img2):
    # Scale the images down if they are too big
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    maxWidth = 1000.0
    if width1 > maxWidth or width2 > maxWidth:
        if width1 > maxWidth and width1 > width2:
            scale = maxWidth / width1
        else:
            scale = maxWidth / width2

        newImg1 = cv2.resize(img1, (int(width1 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
        newImg2 = cv2.resize(img2, (int(width2 * scale), int(height2 * scale)), interpolation = cv2.INTER_AREA)
    else:
        newImg1 = img1.copy()
        newImg2 = img2.copy()

    return (newImg1, newImg2)
```

## Experiments
In the process of improving the project, several experiments were conducted.
One of them was trying to blur the image to remove the background noise. Some of
these blurring techniques were box blue, median blur, gaussian blur, and bilateral blur.
None of these methods improved the features detected, on the contrary it caused the
program to find even less features. Because of the unsatisfying results, the
images are not blurred before using feature detection.

Another experiment consisted of cleaning up the patches containing the differences.
Although the solution finds differences between images, some of the patches can contain
areas which are not different. A possible fix for this was to split the image in
small zones, and then use template matching on each of them to see how different they are.
Unfortunately, even parts which were different were found as being part of the other image.

```python
def cleanPatch(img2, patch):
    pHeight, pWidth = patch.shape[:2]
    oPatch = patch.copy()

    (sHeight, sWidth) = (25, 25)
    if pHeight > sHeight and pWidth > sWidth:
        sX = sY = 0
        while (sY + sHeight < pHeight):
            sPatch = patch[sY : sY + sHeight, sX : sX + sWidth]
            (_, value) = getBestMatch(img2, sPatch)

            if value < 0.25:
                cv2.rectangle(oPatch, (sX, sY), (sX + sWidth, sY + sHeight), (0, 0, 0), 1)

            sX += sWidth
            if sX + sWidth > pWidth:
                sX = 0
                sY += sHeight

            print value
```

## References
[1] R.Szeliski, 'Feature-based alignment' in 'Computer Vision: Algorithms and Applications', 2010, Springer, p. 312
