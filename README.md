# Object detection/recognition using feature based image matching

## Introduction

This repo contains a step-by-step tutorial for doing object detection/recognition using feature detection, description, and matching. This same approach can also be used for e.g. stitching together images, but here the focus is on detection and recognition.

Compared to an approach like neural networks, using a more traditional approach has a couple of nice advantages.

1. You only need one model image per direction you want to recognize an object from. In theory, 6 images should be the maximum required amount for one object.
2. Easier to understand what exactly is happening. Neural networks can be quite black-boxy, while it is much easier to inspect the results of each step of the matching process and see what is(n't) working.
3. Specificity. For tasks where the devil is in the details (e.g. identifying a specific book) these techniques provide quite good results with very little input data.

## Getting started

The example code is written using [Python 3.7](https://www.python.org/downloads/).

For handling data you will need [NumPy](https://pypi.org/project/numpy/).

The main workhorse is [OpenCV](https://opencv.org/). The easiest way to install it is by installing the [unofficial pre-built packages](https://pypi.org/project/opencv-python/) using pip. The only limitation here is that it only contains free algorithms, which leaves out a couple of well known feature detection/description algorithms (more info [here](https://github.com/skvark/opencv-python/issues/126)). But for the purpose of learning, the free algorithms should be enough.

## Code organisation

Implementation can be divided into three rough steps.

1. Load and pre-process the image(s). Feature detection/description is almost always done on gray-scale images, so first you need to load your images in gray-scale. At this point it is also good to do some pre-processing to get rid of noise as well as possible.
2. Detect features and encode their descriptors. If you're going to match two images, you need to figure out some features which can be used for matching. You'll also need to encode a description of these features in a format that makes comparisons possible.
3. Try to match the images. Simply finding the closest matches between individual features isn't difficult. The challenge lies in determining what is considered a good match and filtering out the bad ones. And after that you also need to figure out if all the good matches taken as a whole also match.

The `main.py` file at the root of this repo is for those who want to implement the whole thing for themselves. It only contains some argument parsing for passing in paths to the images you want to compare and for setting the logging level. There is also some simple configuration of the logger. Other than that there is just an empty function with some helpful instructions for how to approach each step and some useful links.

If you instead want to skip one or more steps, the folders `step1`, `step2`, and `step3` contain example implementations of the given step (building on each other). The examples include comments describing the how and why of the implementation and contains links to the relevant resources.

## Useful OpenCV commands

[imread(filename, flags)](https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56): Read image from given file.

[namedWindow(window_name, flags)](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b): Create a window.

[imshow(window_name, image)](https://docs.opencv.org/4.1.0/df/d24/group__highgui__opengl.html#gaae7e90aa3415c68dba22a5ff2cefc25d): Draw an image in a window.

[waitKey(delay)](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7): Wait for a key to be pressed (infinitely if delay = 0).

[destroyAllWindows()](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga6b7fc1c1a8960438156912027b38f481): Does what it says on the can.

[drawKeypoints(image, keypoints, outImage)](https://docs.opencv.org/3.4/d4/d5d/group__features2d__draw.html#gab958f8900dd10f14316521c149a60433): Draw the identified keypoints on top of the chosen image.

[drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg)](https://docs.opencv.org/3.4/d4/d5d/group__features2d__draw.html#ga7421b3941617d7267e3f2311582f49e1): Draw the two images side by side with lines connecting the matching keypoints in each image.
