import argparse
import cv2
from feature_detection_and_description import get_akaze_keypoints_and_descriptors
from image_loading import load_gray_scale_image
import image_matching
import logging
import numpy as np
import sys

def find_model_in_target(path_to_model: str, path_to_target: str):

    logging.debug("find_model_in_target: Called with path_to_model = {} and path_to_target = {}".format(path_to_model, path_to_target))
    
    # STEP 1
    # Regardless of your approach, you first need to load the chosen images from
    # the given files. Most techniques use gray-scale images for detecting
    # features, so you'll probably want to load the images as gray-scale.
    # While loading the images, there are a couple of things you might want to consider:
    # 1. Scale the images up or down. It's a good idea to have some consitency in
    #    the image sizes, so consider scaling the images so e.g. the longer side
    #    is always some specific length. Scaling (and more):
    #    https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    # 2. Consider doing some smoothing of the image. Noise can otherwise cause unwanted
    #    discrepancies when trying to find prominent features. Smoothing with OpenCV:
    #    https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html

    model_image = load_gray_scale_image(path_to_model)
    target_image = load_gray_scale_image(path_to_target)

    # STEP 2
    # Once you have read your images as gray-scale it is time to extract interesting
    # features from the images. Feature extraction has two parts to it: detecting
    # interesting features and describing the area around the feature. For a short
    # introduction to feature detection and descriptors, I recommend this article:
    # https://blog.annaphilips.com/feature-detectors-descriptors-opencv/
    # The article describes the different free alternatives provided by OpenCV.
    # The example code in this project will use the AKAZE detector/descriptor,
    # but feel free to try out other techniques.
    # Useful stuff:
    # https://docs.opencv.org/4.1.0/db/d27/tutorial_py_table_of_contents_feature2d.html

    model_keypoints, model_descriptors = get_akaze_keypoints_and_descriptors(model_image)
    target_keypoints, target_descriptors = get_akaze_keypoints_and_descriptors(target_image)

    # STEP 3
    # With descriptors in hand it is now possible to actually try and match images.
    # Simply matching can be done quite easily using a Brute-Force Matcher
    # (https://docs.opencv.org/4.1.0/dc/dc3/tutorial_py_matcher.html), but if
    # you want to determine whether the images actually match each other, this
    # is the part where you'll have to set the criteria for what constitutes a
    # match. One generally good practice is to filter out matches based on if they
    # are distinct enough. This can be used using ratio test that you can read about
    # here: https://docs.opencv.org/4.1.0/d5/d6f/tutorial_feature_flann_matcher.html.
    # You might also want to check that you don't have duplicate mappings, i.e. two
    # points in one image having the same point in the second image as their best
    # match. In the case of duplicates it's best to just keep the one that is a
    # closer match.
    # A good way to check if the two images actually contain the same object is by
    # using the homography (https://docs.opencv.org/4.1.0/d9/dab/tutorial_homography.html).
    # The basic idea of the homography is to check if there is a way to transform
    # one of the images so that the matching points in each image line up. By setting
    # some limits on how much distortion (scaling and skewing primarily) is acceptable
    # you can then say if the images contain the same object.
    # Useful stuff:
    # https://docs.opencv.org/4.1.0/d1/de0/tutorial_py_feature_homography.html
    # https://docs.opencv.org/4.1.0/d7/dff/tutorial_feature_homography.html

    unfiltered_matches = image_matching.do_2_nn_brute_force_matching_hamming(model_descriptors, target_descriptors)
    ratio_filtered_matches = image_matching.do_2_nn_ratio_filtering(unfiltered_matches)
    duplicate_filtered_matches = image_matching.remove_duplicate_mappings(ratio_filtered_matches)
    homography_filtered_matches = image_matching.filter_with_homography(duplicate_filtered_matches, model_keypoints, target_keypoints)

    draw_matches("Unfiltered matches: {}".format(len(unfiltered_matches)), [m for m, n in unfiltered_matches], target_image, target_keypoints, model_image, model_keypoints)
    draw_matches("Ratio filtered matches: {}".format(len(ratio_filtered_matches)), ratio_filtered_matches, target_image, target_keypoints, model_image, model_keypoints)
    draw_matches("Duplicate filtered matches: {}".format(len(duplicate_filtered_matches)), duplicate_filtered_matches, target_image, target_keypoints, model_image, model_keypoints)
    draw_matches("Homography filtered matches: {}".format(len(homography_filtered_matches)), homography_filtered_matches, target_image, target_keypoints, model_image, model_keypoints)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def draw_matches(window_name, matches, target_image, target_keypoints, model_image, model_keypoints):

    matches_img = np.empty((max(model_image.shape[0], target_image.shape[0]), model_image.shape[1] + target_image.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(target_image, target_keypoints, model_image, model_keypoints, matches, matches_img, flags=cv2.DrawMatchesFlags_DEFAULT)
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, matches_img)


def configure_logging(level: str):
    logging.getLogger().setLevel(getattr(logging, level))
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logging.Formatter("%(asctime)s [%(module)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(streamHandler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--log',
        dest='loglevel',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level')
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
    
    configure_logging(args.loglevel)

    try:
        find_model_in_target(args.model_path, args.target_path)
    except:
        logging.exception('Unexpected exception occured!')