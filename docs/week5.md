# Week 5
Comparing Descriptor Matching Algorithms and Options

## Introduction
This week's blog will contain a comparison between BFMatcher and FLANN. It will also compare the results from changing
some of the parameters in BFMatcher. The descriptors used were generated from the AKAZE algorithm for feature detection
as it provided the best results.

## Brute-Force Matching (BFMatcher)
This method takes each feature from one of the descriptors and compares it to all the other features in the second one.
It returns the matching feature with minimal distance.

## Fast Approximate Nearest Neighbor Search Matching (FLANN)
