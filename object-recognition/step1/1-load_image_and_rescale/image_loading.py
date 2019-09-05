import cv2

MAX_DIMENSION_SIZE = 1024

def load_gray_scale_image(path: str):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = img.shape[:2]
    print("Image path: {}. Original height: {}. Original width: {}.".format(path, original_height, original_width))

    scale_factor = float(MAX_DIMENSION_SIZE) / float(original_width if original_width > original_height else original_width)
    print("Calculated scale factor: {}".format(scale_factor))
    if scale_factor == 1.0:
        return img
        
    interpolation_flag = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation_flag)
    resized_height, resized_width = resized_img.shape[:2]
    print("Resized image dimensions. Height: {}. Width: {}.".format(resized_height, resized_width))

    return resized_img