import cv2

def load_gray_scale_image(path: str):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img