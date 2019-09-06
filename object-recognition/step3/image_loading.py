import cv2
import logging

FILTER_DIAMETER = 9
FILTER_SIGMA_COLOR = 150
FILTER_SIGMA_SPACE = 150
MAX_DIMENSION_SIZE = 1024

def load_gray_scale_image(path: str):

    logging.debug("load_gray_scale_image: Called to load image from path {}".format(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = img.shape[:2]
    logging.debug("Original height: {}. Original width: {}.".format(original_height, original_width))

    # Apply a bilateral filter. Though a little slower than other filters, it helps to keep edges
    # sharp, which helps later in the process.
    # For a fuller explanation see https://docs.opencv.org/4.1.0/d4/d13/tutorial_py_filtering.html and
    # https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    filtered_img = cv2.bilateralFilter(img, FILTER_DIAMETER, FILTER_SIGMA_COLOR, FILTER_SIGMA_SPACE)

    # Calculate how much to scale the image up or down. MAX_DIMENSION_SIZE tells us how long the longest side of
    # the image should be after scaling. This gives us some consistency between images.
    scale_factor = float(MAX_DIMENSION_SIZE) / float(original_width if original_width > original_height else original_width)
    logging.debug("Calculated scale factor: {}".format(scale_factor))
    if scale_factor == 1.0:
        return filtered_img

    # Choose how interpolation should be done for new pixels, depending on if we are scaling up or down.
    # Here we are using bicubic interpolation for zooming and an area based method for shrinking. These 
    # are the given recommendations given here:
    # https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html.
    # For more information about the area based method, here is a good write-up:
    # https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    # The full list of interpolation flags can be found here:
    # https://docs.opencv.org/trunk/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    interpolation_flag = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
    resized_img = cv2.resize(filtered_img, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation_flag)
    resized_height, resized_width = resized_img.shape[:2]
    logging.debug("Resized image dimensions. Height: {}. Width: {}.".format(resized_height, resized_width))

    return resized_img