from __future__ import annotations
import numpy as np
import cv2


# Image I/O 

def read_image(image_path: str) -> np.ndarray:
    """Read an image from *image_path* and return it as a BGR ndarray."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image                              # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure an image has 3 channels (needed before drawing coloured overlays)."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Clip & cast to uint8 in-place; works for float or wider int arrays."""
    return np.clip(image, 0, 255).astype(np.uint8)


#  Chain-code utilities (Freeman 8-connectivity) 

_FREEMAN_DIRS = np.array([
    [0, 1], [-1, 1], [-1, 0], [-1, -1],
    [0, -1], [1, -1], [1, 0], [1, 1],
], dtype=int)                          # directions 0-7


def contour_to_chain_code(contour: np.ndarray) -> list[int]:
    """
    Convert an OpenCV contour (shape N×1×2) to an 8-directional Freeman
    chain code.

    Returns
    -------
    list[int]
        Chain-code values in [0, 7].
    """
    pts = contour[:, 0, :]          # N×2  (x, y)
    chain: list[int] = []
    for i in range(len(pts)):
        dy = pts[(i + 1) % len(pts)][1] - pts[i][1]
        dx = pts[(i + 1) % len(pts)][0] - pts[i][0]
        # clamp to {-1, 0, 1} for robustness
        dy = int(np.sign(dy))
        dx = int(np.sign(dx))
        step = np.array([dy, dx])
        for code, direction in enumerate(_FREEMAN_DIRS):
            if np.array_equal(direction, step):
                chain.append(code)
                break
    return chain


def chain_code_perimeter(chain: list[int]) -> float:
    """
    Estimate perimeter from a Freeman chain code.
    Straight steps (even codes) contribute 1, diagonal (odd) contribute √2.
    """
    straight  = sum(1     for c in chain if c % 2 == 0)
    diagonal  = sum(1     for c in chain if c % 2 == 1)
    return straight + diagonal * np.sqrt(2)


def contour_area(contour: np.ndarray) -> float:
    """Return the area enclosed by an OpenCV contour (px²)."""
    return float(cv2.contourArea(contour))