import cv2
import logging
import math
import numpy as np

MATCH_RATIO_THRESHOLD = 0.8
RANSAC_THRESHOLD = 10.24
MIN_MATCHES_FOR_HOMOGRAPHY = 10
MAX_HOMOGRAPHY_DETERMINANT = 100
MIN_HOMOGRAPHY_DETERMINANT = 0.01
HOMOGRAPHY_SCALE_UPPER_LIMIT = 20
HOMOGRAPHY_SCALE_LOWER_LIMIT = 0.05
HOMOGRAPHY_PERSPECTIVE_LIMIT = 0.0025


def do_2_nn_brute_force_matching_hamming(model_descriptors, target_descriptors):

    # Since AKAZE uses binary string based descriptors, we need to use Hamming distance
    # for matching.
    # https://docs.opencv.org/4.1.0/dc/dc3/tutorial_py_matcher.html
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # We search for the 2 best matches (nearest neighbors) for each point. We can
    # then later filter matches by comparing the 2 best matches to each other.
    matches_2_nn = matcher.knnMatch(target_descriptors, model_descriptors, 2)
    logging.debug("Brute Force Matcher found {} matches".format(len(matches_2_nn)))
    return matches_2_nn


def do_2_nn_ratio_filtering(unfiltered_2_nn_matches):

    logging.debug("Matches before ratio filtering: {}".format(len(unfiltered_2_nn_matches)))
    good_matches = []
    # Filter out matches where the second best result is too close to the best one. If the distances
    # are too close to each other, it is possible that the feature isn't distinct enough to
    # unambiguously match it to a specific feature in the model image.
    # https://docs.opencv.org/4.1.0/d5/d6f/tutorial_feature_flann_matcher.html
    for nearest_neighbour, second_nearest_neigbour in unfiltered_2_nn_matches:
        if nearest_neighbour.distance < MATCH_RATIO_THRESHOLD * second_nearest_neigbour.distance:
            good_matches.append(nearest_neighbour)
    logging.debug("Matches after ratio filtering: {}".format(len(good_matches)))
    return good_matches


def remove_duplicate_mappings(matches):

    # When matching features there is no guarantee that matches will be 1-to-1. There
    # might be several features in the target image that have been matched with the 
    # same feature in the model image. Therefore here we go through all matches and only
    # keep the best match for each model feature.
    logging.debug("Matches before removing duplicates: {}".format(len(matches)))
    best_matches = []
    mapped_model_indeces = []
    match_distances = []

    for match in matches:
        model_idx = match.trainIdx  # Index of the matched keypoint in the model image
        distance = match.distance

        if not model_idx in mapped_model_indeces:
            best_matches.append(match)
            mapped_model_indeces.append(model_idx)
            match_distances.append(distance)
        else:
            idx_of_model_idx = mapped_model_indeces.index(model_idx)
            if distance < match_distances[idx_of_model_idx]:
                best_matches[idx_of_model_idx] = match
                match_distances[idx_of_model_idx] = distance
    logging.debug("Matches after removing duplicates: {}".format(len(best_matches)))

    return best_matches


def filter_with_homography(matches, model_keypoints, target_keypoints):

    # This function is based on the properties of transformation matrices. If you are not
    # familiar or just rusty, I suggest taking a quick look at the following things:
    # https://en.wikipedia.org/wiki/Transformation_matrix
    # https://en.wikipedia.org/wiki/Affine_transformation
    # https://en.wikipedia.org/wiki/3D_projection#Perspective_projection

    # We must require some minimum amount of matches before trying to determine the homography.
    # If you calculate a valid homography with a low number of matches, it is hard to say if
    # it is because the images match or just because of luck (consider that you can calculate
    # it with just 4 points).
    if len(matches) < MIN_MATCHES_FOR_HOMOGRAPHY:
        logging.info("Not enough matches for homography. {} matches given, requires at least {}".format(len(matches), MIN_MATCHES_FOR_HOMOGRAPHY))
        return []

    logging.debug("Matches before homography: {}".format(len(matches)))
    target_keypoint_positions = np.float32([target_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    model_keypoint_positions = np.float32([model_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    filtered_matches = []

    # We use RANSAC (https://en.wikipedia.org/wiki/Random_sample_consensus) to obtain more robust results
    # Docs: https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    homography, mask = cv2.findHomography(target_keypoint_positions, model_keypoint_positions, cv2.RANSAC, RANSAC_THRESHOLD)

    logging.debug("Homography:\n{}".format(homography))
    if homography.size == 0:
        logging.error("Failed to obtain any homography")
        return filtered_matches

    # The following checks are all about how the homography transforms the target image. For us to
    # trust that the homography works because the model has been found in the target image,
    # the homography can't distort the image too much. As an extreme example, if the
    # homography is flattening the image almost down to a line it might be finding a match just
    # because of chance. These checks test different aspects of the transformation.

    # The determinant of the upper left 2 x 2 matrix of the homography tells us (close enough) the
    # total scaling factor of the homography. We don't trust it if the scaling is too much
    # in either direction.
    determinant = (homography[0, 0] * homography[1, 1]) - (homography[0, 1] * homography[1, 0])
    logging.debug("Determinant: {}".format(determinant))
    if determinant > MAX_HOMOGRAPHY_DETERMINANT or determinant < MIN_HOMOGRAPHY_DETERMINANT:
        logging.info("Calculated homography has a determinant outside allowed values")
        return filtered_matches

    # N1 and N2 check how much basis each basis vector (https://en.wikipedia.org/wiki/Basis_(linear_algebra))
    # is scaled during transformation (the columns of the upper left 2 x 2 matrix give the basis vectors; 
    # if you wonder how, see what happens if you multiply (1, 0) and (0, 1) with the matrix). We check this
    # these separately, since the determinant can stay relatively close to 1 if one basis vector is scaled
    # up while the other is scaled down.
    N1 = math.sqrt(math.pow(homography[0, 0], 2) + math.pow(homography[1, 0], 2))
    logging.debug("N1: {}".format(N1))
    if N1 > HOMOGRAPHY_SCALE_UPPER_LIMIT or N1 < HOMOGRAPHY_SCALE_LOWER_LIMIT:
        logging.info("Calculated homography scales x-basis vector beyond trusted limits.")
        return filtered_matches
    N2 = math.sqrt(math.pow(homography[0, 1], 2) + math.pow(homography[1, 1], 2))
    logging.debug("N2: {}".format(N2))
    if N2 > HOMOGRAPHY_SCALE_UPPER_LIMIT or N2 < HOMOGRAPHY_SCALE_LOWER_LIMIT:
        logging.info("Calculated homography scales y-basis vector beyond trusted limits.")
        return filtered_matches

    # For us to trust the homography, it should also have very low levels of perspectivity.
    N3 = math.sqrt(math.pow(homography[2, 0], 2) + math.pow(homography[2, 1], 2))
    logging.debug("N3: {}".format(N3))
    if N3 > HOMOGRAPHY_PERSPECTIVE_LIMIT:
        logging.info("Calculated homography distorts perspective beyond trusted bounds.")
        return filtered_matches

    # Check against mask to only keep matches that weren't filtered out by RANSAC.
    for i, match in enumerate(matches):
        if mask[i] != 0:
            filtered_matches.append(match)
    logging.debug("Matches after homography: {}".format(len(filtered_matches)))

    return filtered_matches
