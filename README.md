# Difference Checker
(c) Vlad Zat, Emmet Doyle, Trent Nguyen, Cristian Anton 2017

# Introduction:
The difference checker is an application created for the purpose of difference that exists in two images that are similar but have fundamental differences. This is achieved through the equalization, image manipulation, morphological applications and template matching to the two images allowing us to produce a final product that highlights the key differences between the one of the images compared to the other.
Tested on OpenCV 3.2.0 with Python 2.7.

## 1. Background Information
Libraries:
CV2 - Extensive computer graphics library with sophisiticated algorithms and functions that can be worked upon to suit the needs of many image/video application.

Numpy and Math - Mathematical libraries that are needed to sort through multidimensional array data and perform equations to correct problems that arise in the application.

EasyGUI - A small graphifics librarie that allows for the quick and easy integration of UI to applications.

## 2. Structure
2.1 Read in images
  - Images are read in through here by manual navigation by the user. The images are stored in image variables for use in the application.
  * EasyGUI's fileopenbox() function was used to allow the user to sift through the files on their PC allowing them to navigate to the image files. Once the images are loaded the button can be pressed and will start the application.

2.2 Image Manipulation and feature detection
  - The images are passed through several functions that downscale, upscale, rotate and transform them so that they are better suited for the operations to detect the differences. During this process the features are detected and stored in an array so that they can be used later in the application as a basis of finding the differences.

2.2.1 Downscale image
  - Big images often cause trouble for applications as they increase run time and worse produce an
  abundant amount of false positives which would otherwise not show up if we downscale the image.
  Here we downscale our image so that we can display our results properly, reduce runtime and noise.

   * In our downscaleImages() we extract the width and height of the images to find out which one of
   the images is bigger. By finding the bigger image we can compared it to the smaller image thus
   creating a scaling factor which we'll use to ensure that the two images are the small dimensions.
   The resizing was done with the cv2.resize() function.

 2.2.2 Rotating the image with feature detection
   - We will try to rotate the second image to match the orientation of the object in the first image. This
   allows our algorithms to perform better as the objects to be detected will be put as close together as they
   can be on the x and y planes.

   * In matchRotation() we first add a border to our second image (i.e. the one to be rotated). This is because
   when we rotate the image we will lose data due to the slight shift which we want to keep until the rotation is
   complete. Next, we put the images through our getMatches() function to get the features that exist in the both
   images. This function uses AKAZE as a feature detector [1] .Using the features we pass the two best lines to
   our getRotationAngle() function and by using the x and y coordinates we calcalate the atan of the two lines
   and apply this angle to the second image. We then rotate the second image using this angle and return the rotated image.

2.2.4 Location Correction
    - The location correction will move the objects within the second image as close as possible to the object in the first using
    the features. As the rotation of the image had been corrected we will simply find the sum needed to be applied to the axis to
    bring the second object to roughly the same position as the first.

   * The coordinates are loaded in the locationCorrection() function. The getMatches() will be used to once more find the differences in the images and using by using the difference we find the sum needed to be transitioned. A translation matrix is created with the x and y axis difference [2]. The warpAffine() function will take this in and apply it to our image.

2.3 Getting the mask
  - The mask of the differences will be populated here. Using histogram equalization techniques, morphological operations and contour
  sizing comparisons we are able to create a mask that display areas that are dense where differences exists.

2.3.1 Creating the mask
   - The initial mask is created with the differences and will be dilated to ensure that positives will cojoin together and be shown as an area of difference. At the same time, false positive which are often isolated will be removed thus creating a mask of areas of difference.
  * In our getMask() function we first use equalizeHist() so that the distribution of pixels is even across the image. By use of our getDifferences() function we populate a newly created image with features that are not present in either images (the differences) as getDifferences() stores the differences in an array. This function use AKAZE for feature detection [1]. Template matching is then applied to every contour seeing if it doesn't exist in the second image. If it did we were able to automate our "dilation" loop which works by drawing a black contour on pixels less than a certain size thus coloring them black while drawing a white contour around pixels that have conjoined. We repeat this until there are no more iterations that can be made. Just using dilation would makes all the pixels bigger. This only increases the biggest differences.

2.4 Applying Template Matching
  - Using the mask attained from 2.3 we are able to draw a contour around all the differences discovered. We apply CLAHE to the images to sharpen the image and by using template matching we can search to see if patches exist on the second image compared to the first. Thus we were able to isolate only the true differences in the two images.

2.4.1 Getting the patches
  - In our getAllPatches() we apply our cv2.boundingRect() to the image where patches exist thus creating an image where every patch, even the noise, is apparent.

2.4.2 Applying CLAHE
  - We now apply Contrasted Limited Adaptive Histogram Equalation to the images thus allowing the distribution to be sharp through the image.
  * In our normaliseImages() function we apply the CLAHE to the entire image rather than just segments of the image. The contrast clipping was set to 40 and the reason for us apply it to the whole image is because we wanted the image to equalize in relation towil itself rather than have sharp points throughout.

2.4.3 Getting the best patches
  - The patches that appear in both and aren't similar to each other are retrieved in this section. This is where only the true difference is retrieved from the application and is drawn to the image.
  * getBestPatches() takes in a list of contours and a theshold. It then uses template matching to find the normalised matched value. It then returns all the contours which are smalled than the threshold. The values for the normalised template matching can be between 0.0 and 1.0 where 1.0 is a perfect match [3] 
  * Our getBestPatchAuto() will try multiple thesholds and determine the lowest theshold where patches are found. This makes sure that only the best patches are returned.
2.5 Displaying the images
  - The contour is applied to this image and then displayed using hstack next to its compared image.

## 3. Extra Notes:

   * While template matching is better at detecting differences, it would take too long to check every pixel in the images.
     This is why the features are filtered using feature detection first.

   * Two types of normalisation are used in the project. While CLAHE normalises the images better, it creates a lot of noise.
     Noise is very detrimental to the feature detection algorithm. This is why equalize hist is used for feature detection.
     When the false-positive contours are eliminated using template matching, CLAHE is used as it makes the images match even more.



##  4. References

  [1] 'AKAZE local features matching', 2014, [Online].
     Available: http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/akaze_matching/akaze_matching.html.
     [Accessed: 2017-10-05]

  [2] R.Szeliski, 'Feature-based alignment' in 'Computer Vision: Algorithms and Applications', 2010, Springer, p. 312

  [3] U.Sinha, 'Template matching', 2010, [Online].
     Available: http://www.aishack.in/tutorials/template-matching.
     [Accessed: 2017-10-26]
     
## Blog
The process of making this project can be found at the following [blog](https://vzat.github.io/comparing_images/).
