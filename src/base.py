"""
Base model for all the filters in the project
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils import to_gray

class Base(ABC):

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Vectorised convolution for a single channel via sliding_window_view."""
        if image.shape >= 3:
            gray_image = to_gray(image)
        else:
            gray_image = image

        kh, kw = kernel.shape
        pad_top, pad_bot = (kh - 1) // 2, kh // 2
        pad_left, pad_right = (kw - 1) // 2, kw // 2

        padded = np.pad(gray_image, ((pad_top, pad_bot), (pad_left, pad_right)), mode='reflect')
        windows = sliding_window_view(padded, (kh, kw))          # view, no copy
        output = np.sum(windows * kernel.reshape(1, 1, kh, kw),  # broadcast
                        axis=(2, 3))
        return np.clip(output, 0, 255)
    