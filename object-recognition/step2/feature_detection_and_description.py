import cv2
import logging

AKAZE_RESPONSE_THRESHOLD = 0.005
AKAZE_OCTAVES = 4
AKAZE_OCTAVE_LAYERS = 11

def get_akaze_keypoints_and_descriptors(img):

    akaze = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB, 0, 3, AKAZE_RESPONSE_THRESHOLD, AKAZE_OCTAVES, AKAZE_OCTAVE_LAYERS)
    keypoints, descriptors = akaze.detectAndCompute(img, None)
    return keypoints, descriptors