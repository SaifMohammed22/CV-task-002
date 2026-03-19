"""
Helper functions
"""

import numpy as np
import cv2

def read_image(image_path):
    """Read image and convert it to np.ndarray"""
    return cv2.imread(image_path)

def save_image(output_path, image: np.ndarray):
    """Write image to path"""
    return cv2.imwrite(output_path, image)

def to_gray(image: np.ndarray):
    """Convert an image to gray scale"""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

