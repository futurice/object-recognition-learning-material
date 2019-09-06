import cv2
import logging
import math
import numpy as np

MATCH_RATIO_THRESHOLD = 0.8
RANSAC_THRESHOLD = 10.24
MAX_HOMOGRAPHY_DETERMINANT = 20
MIN_HOMOGRAPHY_DETERMINANT = 0.05
HOMOGRAPHY_SCALE_UPPER_LIMIT = 15
HOMOGRAPHY_SCALE_LOWER_LIMIT = 0.05
HOMOGRAPHY_PERSPECTIVE_LIMIT = 0.0025

def do_2_nn_brute_force_matching_hamming(model_descriptors, target_descriptors):

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches_2_nn = matcher.knnMatch(target_descriptors, model_descriptors, 2)
    return matches_2_nn

def do_2_nn_ratio_filtering(unfiltered_2_nn_matches):

    good_matches = []
    for nearest_neighbour, second_nearest_neigbour in unfiltered_2_nn_matches:
        if nearest_neighbour.distance < MATCH_RATIO_THRESHOLD * second_nearest_neigbour.distance:
            good_matches.append(nearest_neighbour)
    return good_matches

def remove_duplicate_mappings(matches):

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

    return best_matches

def filter_with_homography(matches, model_keypoints, target_keypoints):

    target_keypoint_positions = np.float32([target_keypoints[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    model_keypoint_positions = np.float32([model_keypoints[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    filtered_matches = []

    homography, mask = cv2.findHomography(target_keypoint_positions, model_keypoint_positions, cv2.RANSAC, RANSAC_THRESHOLD)

    if not homography or homography.shape[0] < 3 or homography.shape[1] < 3:
        return filtered_matches

    determinant = (homography[0, 0] * homography[1, 1]) - (homography[0, 1] * homography[1, 0])

    if determinant > MAX_HOMOGRAPHY_DETERMINANT or determinant < MIN_HOMOGRAPHY_DETERMINANT:
        return filtered_matches

    N1 = math.sqrt(math.pow(homography[0, 0], 2) + math.pow(homography[1, 0], 2))
    if N1 > HOMOGRAPHY_SCALE_UPPER_LIMIT or N1 < HOMOGRAPHY_SCALE_LOWER_LIMIT:
        return filtered_matches

    N2 = math.sqrt(math.pow(homography[0, 1], 2) + math.pow(homography[1, 1], 2))
    if N2 > HOMOGRAPHY_SCALE_UPPER_LIMIT or N2 < HOMOGRAPHY_SCALE_LOWER_LIMIT:
        return filtered_matches

    N3 = math.sqrt(math.pow(homography[2, 0], 2) + math.pow(homography[2, 1], 2))
    if N3 > HOMOGRAPHY_PERSPECTIVE_LIMIT:
        return filtered_matches

    for i, match in enumerate(matches):
        if mask[i] != 0:
            filtered_matches.append(match)

    return filtered_matches
