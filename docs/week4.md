# Week 4
Comparing different feature detection algorithms

1. FAST - Features from Accelerated Segment Test [1]

This is the FAST feature detection algorithm with the Non-Maximum Suppression turned on. It detect many similarities between the images including the background.
This algorithms will probably not be used because the FAST object in the feature2d module cannot create a descriptor for images.

| PCB1 | PCB2 |
| :---: | :---: |
| <img src="images/fast1.jpg" width="300"> | <img src="images/fast2.jpg" width="300"> |


## References:
[1] http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
