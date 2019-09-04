# Object detection/recognition using feature based image matching

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

On a high level, implementation is separated into three steps which build on each other.

TODO: Add better description once the code organisation becomes clear.
