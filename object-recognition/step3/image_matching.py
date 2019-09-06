import cv2
import logging

MATCH_RATIO_THRESHOLD = 0.8

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

