
from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils import to_gray


class Base(ABC):
    """Abstract base class that every filter must inherit from."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to the given image and return the result."""

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Vectorised convolution for a single-channel image via sliding_window_view."""
        gray_image = to_gray(image) if len(image.shape) >= 3 else image

        kh, kw = kernel.shape
        pad_top,  pad_bot   = (kh - 1) // 2, kh // 2
        pad_left, pad_right = (kw - 1) // 2, kw // 2

        padded  = np.pad(gray_image,
                         ((pad_top, pad_bot), (pad_left, pad_right)),
                         mode='reflect')
        windows = sliding_window_view(padded, (kh, kw))          # zero-copy view
        output  = np.einsum('ijkl,kl->ij', windows, kernel)      # fast contraction
        return np.clip(output, 0, 255).astype(np.uint8)