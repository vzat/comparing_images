# Week 3
Researching methods for comparing images

## Methods for comparing images

* Structural Similarity Index (SSIM)

  This method compares the perceived structural similarity between images. It compares parts of the image instead of comparing the whole image. When compared to other comparing methods such as Mean Squared Error (MSE) which only estimates the errors in the images.[1]

  An implementation on this method can be found in the 'scikit-image' collection, in the module 'measure'. It can also be implemented using OpenCV by applying the SSIM algorithm directly.

* Histogram Comparison

  Histograms are a way of visualising change over time of a particular object or system. The Histogram Comparison method compares histograms generated from the pixels of each image and compares them. It can detect which parts of the image are the same and which are different. This methods works because humans use colours to recognise objects. [2]

  The library 'numpy' can generate histograms which then can be compared to spot the differences between images.

* Local Invariant Feature Detection

  Similar to the SSIM method, the Local Invariant Feture Detection method finds Local Features of a image and compares them to it's neighbours. Multiple properties can be compared such as intensity, colour and texture. [3]

  There are multiple algoithms for Feature Detection such as Corner Detection, SIFT, SURF, FAST or ORB. ORB (Oriented FAST and Rotated BRIEF) is a free alternative solution made by the developers of OpenCV to other algorithms which are patented. [4]

## Choosing a method

The most promising method from the ones detailed above is the Feature Detection method. This provides the most robust algorithm for detecting differences between images without sacrificing too much of the performance. From the algorithms that use this method, ORB seems to be best choice as it was designed to work with OpenCV.


## References
[1] A.Rosebrock, 'Mean Squared Error vs. Structural Similarity Measure' in 'How-To: Python Compare Two Images', 2014, [Online]. Available: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/. [Accessed: 2017-09-28]

[2] M.Patacchiola, 'The Simplest Classifier: Histogram Comparison', 2016, [Online]. Available: https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html. [Accessed: 2017-09-28]

[3] T.Tuytelaars and K.Mikolajczyk, 'Local Invariant Feature Detectors: A Survey' from 'Foundations and Trends' in 'Computer Graphics and Vision', Vol. 3, No. 3, 2007 177–280

[4] 'Feature Detection and Description', 2014, [Online]. Available: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html. [Accessed: 2017-09-28]
