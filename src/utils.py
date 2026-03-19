"""
Helper functions
"""

import numpy as np
import cv2

def read_image(image_path):
    """Read image and convert it to np.ndarray"""
    return cv2.imread(image_path, 0)

def to_gray(image: np.ndarray):
    """Convert an image to gray scale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
