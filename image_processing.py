import cv2
import numpy as np
from skimage import io, color, filters, exposure

def load_image(path):
    return io.imread(path)

def to_grayscale(image):
    if len(image.shape) == 3:
        return color.rgb2gray(image)
    return image

def enhance_contrast(image):
    return exposure.equalize_adapthist(image)

def apply_filter(image, filter_name):
    if filter_name == "sobel":
        return filters.sobel(image)
    elif filter_name == "gaussian":
        return filters.gaussian(image, sigma=1)
    elif filter_name == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0,-1,0]])
        return cv2.filter2D((image*255).astype(np.uint8), -1, kernel)
    else:
        return image
