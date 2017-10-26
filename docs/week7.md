# Week 7
Testing Template Matching

## Introduction
This week's blog is going to talk about what Template Matching is and how it can be used to detect differences. In last week's blog,
multiple areas of the screen have been choosen as possible candidates for containing a difference between the images. To further eliminate them,
Template Matching is going to be applied for each area found.

## Template Matching
Template Matching compares a patch from an image to another image to find how similar they are [1]. OpenCV implements this in the `cv2.matchTemplate()` function. It supports six types of comparisons: `cv2.TM_CCOEFF`, `cv2.TM_CCORR`, `cv2.TM_SQDIFF`, `cv2.TM_CCOEFF_NORMED`, `cv2.TM_CCORR_NORMED`, `cv2.TM_SQDIFF_NORMED`, where the last three are just normalised versions of the first three. `cv2.TM_CCOEFF` and `cv2.TM_CCORR` are correlation based algorithms in which the best normalised result is 1.0 and `cv2.TM_SQDIFF` is difference based where the best normalised result is 0.0 [2]. The following code draws a rectangle where the best match for the patch was found and prints in the terminal the calculated values.

### Code
```python
def findBestMatch(window, img, patch, method, maxLoc = True):
    patchSize = patch.shape

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(image = gImg, templ = gPatch, method = method)

    # Get the maximum location for CCOEFF and CCORR otherwise get the minimum location
    if maxLoc:
        (_, value, _, (x, y)) = cv2.minMaxLoc(result)
    else:
        (value, _, (x, y), _) = cv2.minMaxLoc(result)

    print window, value

    cv2.rectangle(img, (x, y), (x + patchSize[1], y + patchSize[0]), (255, 0, 0), 3)

    cv2.imshow('Template Matching', img)
    cv2.waitKey(0)
```

## Testing Template Matching
During testing the unnormalised version of `cv2.TM_CCORR` provided consistent bad results. This combined with the fact that the normalised version of the images give results that are easier to use and understand, only the normalised versions of the methods are going to be used. The following patch from the `pcb1.jpg` image was used:

| Patch |
| :---: |
| <img src="images/diffsNoisy.jpg" width="600"> |

### First Image
These are the best matches the algorithms have found for the patch in `pcb1.jpg`

| PCB1 - cv2.TM_CCOEFF_NORMED | PCB1 - cv2.TM_CCORR_NORMED | PCB1 - cv2.TM_SQDIFF_NORMED |
| :---: | :---: | :---: |
| <img src="images/pcb1_cv2.TM_CCOEFF_NORMED" width="200"> | <img src="images/pcb1_cv2.TM_CCORR_NORMED" width="200"> | <img src="images/pcb1_cv2.TM_SQDIFF_NORMED" width="200"> |  

The best match for each algorithm resulted in the following values:

| Method | Value |
| :---: | :---: |
| cv2.TM_CCOEFF_NORMED | 0.999999821186 |
| cv2.TM_CCORR_NORMED | 0.999999940395 |
| cv2.TM_SQDIFF_NORMED | 1.33158494009e-07 |

Besides `cv2.TM_SQDIFF`, the other two algorithm resulted in values close to 1.0.

### Second Image
These are the best matches the algorithms have found for the same patch in `pcb2.jpg`

| PCB2 - cv2.TM_CCOEFF_NORMED | PCB2 - cv2.TM_CCORR_NORMED | PCB2 - cv2.TM_SQDIFF_NORMED |
| :---: | :---: | :---: |
| <img src="images/pcb2_cv2.TM_CCOEFF_NORMED" width="200"> | <img src="images/pcb2_cv2.TM_CCORR_NORMED" width="200"> | <img src="images/pcb2_cv2.TM_SQDIFF_NORMED" width="200"> |  

The best match for each algorithm resulted in the following values:

| Method | Value |
| :---: | :---: |
| cv2.TM_CCOEFF_NORMED | 0.282997101545 |
| cv2.TM_CCORR_NORMED | 0.79689925909 |
| cv2.TM_SQDIFF_NORMED | 0.447067975998 |

Besides the first methods, the other methods have failed to found patch which is expected as it doesn't contain many similar features. Even so, from the values it can be seen `cv2.TM_COEFF` has the worst normalised value even tough it has correctly identified the image.

## Finding the differences
To verify if a patch is different the coordinates of the best match is compared to the coordinates of the original patch. If the coordinates are different then it hasn't found a match. Otherwise a threshold (such as 0.5) is used, the areas that have a low match can be considered as containing a difference while if it's above the treshold then that area was found correctly in the second image.

## Conclusion
From the above tests, it can be observed that `cv2.TM_COEFF` found the correct area of the image even with only a fraction of the patch being the same. Even so, the value resulted is very low which means is not even close to a perfect match. This requires further testing with the other areas found in the previous week. Also, Template Matching has the problem that it can only find a match if the features in both images are the same size and orientation [3]. A solution for this is currently being implemented by the other team members.

## References
[1] OpenCV, 'Template Matching', 2017, [Online]. Available: https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html. [Accessed: 2017-10-26]

[2] U.Sinha, 'Template matching', 2010, [Online]. Available: http://www.aishack.in/tutorials/template-matching. [Accessed: 2017-10-26]

[3] A.Rosebrock, 'Multi-scale Template Matching using Python and OpenCV', 2015, [Online]. Available: https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/. [Accessed: 2017-10-26]
