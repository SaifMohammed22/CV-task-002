"""
Implementation of Gaussian and Prewitt filters
"""

from .base import Base
import numpy as np
import cv2

class GaussianFilter(Base):
    """7x7 Gaussian filter for smoothing"""
    @property
    def kernel(self):
        return (1.0 / 140.0) * np.array([[1, 1, 2, 2, 2, 1, 1],
                                         [1, 2, 2, 4, 2, 2, 1],
                                         [2, 2, 4, 8, 4, 2, 2],
                                         [2, 4, 8, 16, 8, 4, 2],
                                         [2, 2, 4, 8, 4, 2, 2],
                                         [1, 2, 2, 4, 2, 2, 1],
                                         [1, 1, 2, 2, 2, 1, 1]])
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._convolve(image, self.kernel)



class PrewittFilter(Base):
    """X and Y Prewitt filters """

    @property
    def kernelX(self):
        return (1.0 / 3.0) * np.array([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]])
    @property
    def kernelY(self):
        return (1.0 / 3.0) * np.array([[1, 1, 1],
                                       [0, 0, 0],
                                       [-1, -1, -1]])
    
    def apply(self, image):
        horizontal = self._convolve(image, self.kernelX)
        vertical = self._convolve(image, self.kernelY)

        return horizontal, vertical

class CanyFilter(Base):
    """Cany edge Detector"""
    def __init__(self):
        self.prewitt = PrewittFilter()

    def apply(self, image):
        pass

    def getGradients(self, image):
        return self.prewitt.apply(image)

    def getMagnitude(self, GX, GY):
        mag = np.sqrt(GX ** 2 + GY ** 2) / 1.4142
        return mag

    def getAngle(self, GX, GY):
        return np.atan(GX / (GY + 0.0001))