import cv2
import logging
import math
import numpy as np

MATCH_RATIO_THRESHOLD = 0.8
RANSAC_THRESHOLD = 10.24
MIN_MATCHES_FOR_HOMOGRAPHY = 10
MAX_HOMOGRAPHY_DETERMINANT = 20
MIN_HOMOGRAPHY_DETERMINANT = 0.05
HOMOGRAPHY_SCALE_UPPER_LIMIT = 15
HOMOGRAPHY_SCALE_LOWER_LIMIT = 0.05
HOMOGRAPHY_PERSPECTIVE_LIMIT = 0.0025

def do_2_nn_brute_force_matching_hamming(model_descriptors, target_descriptors):

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches_2_nn = matcher.knnMatch(target_descriptors, model_descriptors, 2)
    logging.debug("Brute Force Matcher found {} matches".format(len(matches_2_nn)))
    return matches_2_nn


def do_2_nn_ratio_filtering(unfiltered_2_nn_matches):

    logging.debug("Matches before ratio filtering: {}".format(len(unfiltered_2_nn_matches)))
    good_matches = []
    for nearest_neighbour, second_nearest_neigbour in unfiltered_2_nn_matches:
        if nearest_neighbour.distance < MATCH_RATIO_THRESHOLD * second_nearest_neigbour.distance:
            good_matches.append(nearest_neighbour)
    logging.debug("Matches after ratio filtering: {}".format(len(good_matches)))
    return good_matches


def remove_duplicate_mappings(matches):

    logging.debug("Matches before removing duplicates: {}".format(len(matches)))
    best_matches = []
    mapped_model_indeces = []
    match_distances = []

    for match in matches:
        model_idx = match.trainIdx
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

    if len(matches) < MIN_MATCHES_FOR_HOMOGRAPHY:
        logger.info("Not enough matches for homography. {} matches given, requires at least {}".format(len(matches), MIN_MATCHES_FOR_HOMOGRAPHY))
        return []

    logging.debug("Matches before homography: {}".format(len(matches)))
    target_keypoint_positions = np.float32([target_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    model_keypoint_positions = np.float32([model_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    filtered_matches = []

    homography, mask = cv2.findHomography(target_keypoint_positions, model_keypoint_positions, cv2.RANSAC, RANSAC_THRESHOLD)

    if homography is None:
        logging.error("Failed to obtain any homography")
        return filtered_matches

    determinant = (homography[0, 0] * homography[1, 1]) - (homography[0, 1] * homography[1, 0])

    if determinant > MAX_HOMOGRAPHY_DETERMINANT or determinant < MIN_HOMOGRAPHY_DETERMINANT:
        logging.info("Calculated homography has a determinant outside allowed values")
        return filtered_matches

    N1 = math.sqrt(math.pow(homography[0, 0], 2) + math.pow(homography[1, 0], 2))
    if N1 > HOMOGRAPHY_SCALE_UPPER_LIMIT or N1 < HOMOGRAPHY_SCALE_LOWER_LIMIT:
        logging.info("Calculated homography scales image beyond trusted limits.")
        return filtered_matches

    N2 = math.sqrt(math.pow(homography[0, 1], 2) + math.pow(homography[1, 1], 2))
    if N2 > HOMOGRAPHY_SCALE_UPPER_LIMIT or N2 < HOMOGRAPHY_SCALE_LOWER_LIMIT:
        logging.info("Calculated homography scales image beyond trusted limits.")
        return filtered_matches

    N3 = math.sqrt(math.pow(homography[2, 0], 2) + math.pow(homography[2, 1], 2))
    if N3 > HOMOGRAPHY_PERSPECTIVE_LIMIT:
        logging.info("Calculated homography distorts perspective beyond trusted bounds.")
        return filtered_matches

    for i, match in enumerate(matches):
        if mask[i] != 0:
            filtered_matches.append(match)
    logging.debug("Matches after homography: {}".format(len(filtered_matches)))

    return filtered_matches
