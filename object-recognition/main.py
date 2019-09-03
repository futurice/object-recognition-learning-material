import argparse
import logging

logger = logging.getLogger()

def find_model_in_target(path_to_model: str, path_to_target: str):
    
    # STEP 1
    # Regardless of your approach, you first need to load the chosen images from
    # the given files. Most techniques use gray-scale images for detecting
    # features, so you'll probably want to load the images as gray-scale.
    # While loading the images, there are a couple of things you might want to consider:
    # 1. Consider doing some smoothing of the image. Noise can otherwise cause unwanted
    #    discrepancies when trying to find prominent features. Smoothing with OpenCV:
    #    https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    # 2. Scale the images up or down. It's a good idea to have some consitency in
    #    the image sizes, so consider scaling the images so e.g. the longer side
    #    is always some specific length. Scaling (and more):
    #    https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html

    # STEP 2
    # Once you have read your images as gray-scale it is time to extract interesting
    # features from the images. Feature extraction has two parts to it: detecting
    # interesting features and describing the area around the feature. For a short
    # introduction to feature detection and descriptors, I recommend this article:
    # https://blog.annaphilips.com/feature-detectors-descriptors-opencv/
    # The article describes the different free alternatives provided by OpenCV.
    # The example code in this project will use the AKAZE detector/descriptor,
    # but feel free to try out other techniques.

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model-path',
        dest='model_path',
        type=str,
        help='Path to the model image that you are trying to find in the target image.'
    )
    parser.add_argument(
        '-t',
        '--target-path',
        dest='target_path',
        type=str,
        help='Path of the target image in which you are trying to find an object.'
    )
    args = parser.parse_args()

    try:
        find_model_in_target(args.model_path, args.target_path)
    except:
        logger.exception('Unexpected exception occured!')