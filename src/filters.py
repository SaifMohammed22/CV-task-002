from __future__ import annotations

import numpy as np
import cv2

from .base import Base
from .utils import to_gray, to_bgr, ensure_uint8



# 1. Gaussian smoothing 

class GaussianFilter(Base):
    """7×7 Gaussian filter for image smoothing."""

    @property
    def kernel(self) -> np.ndarray:
        return (1.0 / 140.0) * np.array([
            [1, 1, 2, 2, 2, 1, 1],
            [1, 2, 2, 4, 2, 2, 1],
            [2, 2, 4, 8, 4, 2, 2],
            [2, 4, 8, 16, 8, 4, 2],
            [2, 2, 4, 8, 4, 2, 2],
            [1, 2, 2, 4, 2, 2, 1],
            [1, 1, 2, 2, 2, 1, 1],
        ], dtype=np.float64)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._convolve(image, self.kernel)


# 2. Prewitt edge filter 

class PrewittFilter(Base):
    """Prewitt edge detector – returns gradient magnitude as uint8."""

    @property
    def kernel_x(self) -> np.ndarray:
        # No 1/3 scaling: keeps gradient magnitudes in the full [−255, 255] range.
        return np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ], dtype=np.float64)

    @property
    def kernel_y(self) -> np.ndarray:
        return np.array([
            [ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1, -1],
        ], dtype=np.float64)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return gradient-magnitude image (uint8)."""
        gx = self._convolve_signed(image, self.kernel_x)
        gy = self._convolve_signed(image, self.kernel_y)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return ensure_uint8(magnitude)

    def apply_xy(self, image: np.ndarray):
        """Return (horizontal_edges, vertical_edges) as signed float64 arrays."""
        return (
            self._convolve_signed(image, self.kernel_x),
            self._convolve_signed(image, self.kernel_y),
        )


# 3. Canny edge detector 

class CannyFilter(Base):
    """
    Canny edge detector built from scratch.
    It reuses the existing GaussianFilter and PrewittFilter for steps 1 and 2.
    """

    def __init__(self, low: int = 50, high: int = 150) -> None:
        self.low = low
        self.high = high
        # Instantiate your existing filters to avoid rewriting code!
        self.gaussian = GaussianFilter()
        self.prewitt = PrewittFilter()

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Step 1: Noise Reduction
        # We use your GaussianFilter to blur the image. 
        # (It automatically handles grayscale conversion inside _convolve)
        smoothed = self.gaussian.apply(image)

        # Step 2: Gradient Calculation
        # We use your PrewittFilter to get the X and Y gradients
        gx, gy = self.prewitt.apply_xy(smoothed)
        
        # Convert to float64 to prevent overflow during math operations
        gx = gx.astype(np.float64)
        gy = gy.astype(np.float64)

        # Calculate Magnitude (edge strength) and Direction (edge angle)
        magnitude = np.hypot(gx, gy)
        
        # Calculate angle in degrees (0 to 180)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction[direction < 0] += 180

        # Step 3: Non-Maximum Suppression
        # Thins the edges down to 1-pixel width
        nms_image = self._non_max_suppression(magnitude, direction)

        # Step 4 & 5: Double Thresholding and Hysteresis
        # Identifies strong/weak edges and links them
        final_edges = self._hysteresis(nms_image)

        return ensure_uint8(final_edges)

    def _non_max_suppression(self, mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """
        Suppresses non-maximum pixels to make edges exactly 1 pixel wide.
        """
        h, w = mag.shape
        out = np.zeros((h, w), dtype=np.float64)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                q = 255
                r = 255

                # Discretize the angle into 4 main directions: 0, 45, 90, 135
                
                # Direction: 0 degrees (Horizontal edge, compare North-South)
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = mag[i, j + 1]
                    r = mag[i, j - 1]
                
                # Direction: 45 degrees (Diagonal)
                elif (22.5 <= angle[i, j] < 67.5):
                    q = mag[i + 1, j - 1]
                    r = mag[i - 1, j + 1]
                
                # Direction: 90 degrees (Vertical edge, compare East-West)
                elif (67.5 <= angle[i, j] < 112.5):
                    q = mag[i + 1, j]
                    r = mag[i - 1, j]
                
                # Direction: 135 degrees (Diagonal)
                elif (112.5 <= angle[i, j] < 157.5):
                    q = mag[i - 1, j - 1]
                    r = mag[i + 1, j + 1]

                # Keep the pixel only if it is the local maximum in its direction
                if mag[i, j] >= q and mag[i, j] >= r:
                    out[i, j] = mag[i, j]
                else:
                    out[i, j] = 0

        return out

    def _hysteresis(self, img: np.ndarray) -> np.ndarray:
        """
        Links weak edges to strong edges. Weak edges not connected to strong edges are removed.
        """
        # Dynamically scale thresholds based on the maximum pixel intensity found
        max_val = img.max() if img.max() > 0 else 1
        high_th = max_val * (self.high / 255.0)
        low_th = high_th * (self.low / self.high) if self.high > 0 else 0

        h, w = img.shape
        res = np.zeros((h, w), dtype=np.uint8)

        strong = 255
        weak = 75

        # Find coordinates of strong and weak pixels
        strong_i, strong_j = np.where(img >= high_th)
        weak_i, weak_j = np.where((img <= high_th) & (img >= low_th))

        # Assign values to the result image
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        # Hysteresis Tracking: Check 8-connected neighbors for weak pixels
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if res[i, j] == weak:
                    # If a strong pixel is adjacent to this weak pixel, keep it as a strong edge
                    if strong in res[i - 1 : i + 2, j - 1 : j + 2]:
                        res[i, j] = strong
                    else:
                        # Otherwise, discard it as noise
                        res[i, j] = 0

        return res

# 4. Hough line detector 

class HoughLinesFilter(Base):
    """
    Detects straight lines with the Probabilistic Hough Transform and
    superimposes them on the original image.
    """

    def __init__(
        self,
        rho: float = 1,
        theta: float = np.pi / 180,
        threshold: int = 80,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        self._canny           = CannyFilter(canny_low, canny_high)
        self.rho              = rho
        self.theta            = theta
        self.threshold        = threshold
        self.min_line_length  = min_line_length
        self.max_line_gap     = max_line_gap

    def apply(self, image: np.ndarray) -> np.ndarray:
        edges  = self._canny.apply(image)
        output = to_bgr(image)
        lines  = cv2.HoughLinesP(
            edges,
            self.rho, self.theta, self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return output


# 5. Hough circle detector 

class HoughCirclesFilter(Base):
    """
    Detects circles with the Hough gradient method and
    superimposes them on the original image.
    """

    def __init__(
        self,
        dp: float = 1.2,
        min_dist: int = 30,
        param1: int = 100,
        param2: int = 30,
        min_radius: int = 0,
        max_radius: int = 0,
    ) -> None:
        self.dp         = dp
        self.min_dist   = min_dist
        self.param1     = param1
        self.param2     = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray   = to_gray(image)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        output = to_bgr(image)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is not None:
            for cx, cy, r in np.round(circles[0]).astype(int):
                cv2.circle(output, (cx, cy), r,  (0, 255, 0), 2)
                cv2.circle(output, (cx, cy), 2,  (0,   0, 255), 3)
        return output


# 6. Hough ellipse detector 

class HoughEllipsesFilter(Base):
    """
    Fits ellipses to contours found after Canny edge detection and
    superimposes them on the original image.

    OpenCV does not expose a direct Hough-ellipse transform; the standard
    approach is contour-fitting, which is equivalent for typical use cases.
    """

    def __init__(
        self,
        canny_low: int  = 50,
        canny_high: int = 150,
        min_points: int = 5,
    ) -> None:
        self._canny     = CannyFilter(canny_low, canny_high)
        self.min_points = min_points

    def apply(self, image: np.ndarray) -> np.ndarray:
        edges   = self._canny.apply(image)
        output  = to_bgr(image)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        for cnt in contours:
            if len(cnt) < self.min_points:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(output, ellipse, (255, 0, 0), 2)
            except cv2.error:
                # fitEllipse can fail on collinear / degenerate point sets
                continue
        return output


# 7. Active Contour (Snake) 

class ActiveContourFilter(Base):
    """
    Greedy Active Contour (snake) implementation.

    The snake is initialised as an ellipse centred on the image.  Each
    iteration every point moves to the neighbour (in a (2w+1)×(2w+1) window)
    that minimises the combined internal + external energy.

    Returns the BGR image with the evolved contour drawn in green, together
    with the chain code, perimeter, and area as metadata.
    """

    def __init__(
        self,
        n_points:   int   = 200,
        alpha:      float = 0.01,   # elasticity  (internal – continuity)
        beta:       float = 0.1,    # stiffness   (internal – curvature)
        gamma:      float = 0.01,   # step size
        iterations: int   = 200,
        w_size:     int   = 3,
    ) -> None:
        self.n_points   = n_points
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.iterations = iterations
        self.w_size     = w_size

    # helpers 

    @staticmethod
    def _init_ellipse(h: int, w: int, n: int) -> np.ndarray:
        """Return *n* equidistant points on the ellipse centred in the image."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = w / 2 + (w / 2 - 10) * np.cos(t)
        y = h / 2 + (h / 2 - 10) * np.sin(t)
        return np.column_stack([x, y]).astype(np.float64)

    @staticmethod
    def _external_energy(gray: np.ndarray) -> np.ndarray:
        """Negative gradient magnitude as external energy map."""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return -(gx ** 2 + gy ** 2)

    def _internal_energy(self, snake: np.ndarray, i: int) -> float:
        """Continuity + curvature energy for the *i*-th point."""
        n   = len(snake)
        p_prev = snake[(i - 1) % n]
        p_curr = snake[i]
        p_next = snake[(i + 1) % n]

        # mean distance for normalisation
        d_mean = np.mean(np.linalg.norm(np.diff(snake, axis=0), axis=1))

        continuity = (d_mean - np.linalg.norm(p_curr - p_prev)) ** 2
        curvature  = np.linalg.norm(p_next - 2 * p_curr + p_prev) ** 2

        return self.alpha * continuity + self.beta * curvature

    # main apply 

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Evolve the snake and return the annotated BGR image.

        Extra attributes set on return
        ──────────────────────────────
        self.chain_code  – Freeman-8 chain code (list[int])
        self.perimeter   – estimated perimeter (float)
        self.area        – enclosed area (float)
        self.contour     – final snake points as (N,1,2) int32 array
        """
        gray   = to_gray(image)
        h, w   = gray.shape
        energy = self._external_energy(gray)

        snake = self._init_ellipse(h, w, self.n_points)
        r     = self.w_size                    # search radius

        for _ in range(self.iterations):
            for i in range(len(snake)):
                original  = snake[i].copy()
                best_e    = np.inf
                best_pos  = original.copy()

                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx = int(original[0] + dx)
                        ny = int(original[1] + dy)
                        if not (0 <= nx < w and 0 <= ny < h):
                            continue
                        # temporarily place point to evaluate internal energy
                        snake[i] = [nx, ny]
                        e = (self._internal_energy(snake, i)
                             + self.gamma * energy[ny, nx])
                        if e < best_e:
                            best_e  = e
                            best_pos = np.array([nx, ny], dtype=np.float64)

                snake[i] = best_pos  # commit the best candidate

        #  build OpenCV contour for chain-code + area utils 
        self.contour = snake.astype(np.int32).reshape(-1, 1, 2)

        from utils import contour_to_chain_code, chain_code_perimeter, contour_area
        self.chain_code = contour_to_chain_code(self.contour)
        self.perimeter  = chain_code_perimeter(self.chain_code)
        self.area       = contour_area(self.contour)

        # draw result 
        output = to_bgr(image)
        cv2.polylines(output, [self.contour], isClosed=True, color=(0, 255, 0), thickness=2)
        return output