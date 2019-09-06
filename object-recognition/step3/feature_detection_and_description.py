import cv2
import logging

# Feel free to play around with these values and see how it affects the results.
AKAZE_RESPONSE_THRESHOLD = 0.005
AKAZE_OCTAVES = 4
AKAZE_OCTAVE_LAYERS = 11

def get_akaze_keypoints_and_descriptors(img):

    # There is no short and easy way to describe AKAZE. If you really want to understand it, the paper
    # can be found here: http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf.
    # A very condensed description can be found here: http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html
    # AKAZE OpenCV refrence: https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html
    logging.debug("Creating AKAZE with threshold {}, {} octaves, and {} octave layers".format(
        AKAZE_RESPONSE_THRESHOLD, AKAZE_OCTAVES, AKAZE_OCTAVE_LAYERS
    ))
    akaze = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB, 0, 3, AKAZE_RESPONSE_THRESHOLD, AKAZE_OCTAVES, AKAZE_OCTAVE_LAYERS)
    # Keypoints and descriptors can be calculated separately, but since they both require the same initial calculations
    # doing it in one go saves time.
    keypoints, descriptors = akaze.detectAndCompute(img, None)
    return keypoints, descriptors