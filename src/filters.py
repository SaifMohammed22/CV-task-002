from __future__ import annotations

import numpy as np
import cv2

from .base import Base
from .utils import (
    to_gray,
    to_bgr,
    ensure_uint8,
    contour_to_chain_code,
    chain_code_perimeter,
    contour_area,
)



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


# 2. Sobel edge filter

class SobelFilter(Base):
    """Sobel edge detector - returns gradient magnitude as uint8."""

    @property
    def kernel_x(self) -> np.ndarray:
        # No 1/8 scaling: keeps gradient magnitudes in a wide dynamic range.
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=np.float64)

    @property
    def kernel_y(self) -> np.ndarray:
        return np.array([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1],
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
    It reuses the existing GaussianFilter and SobelFilter for steps 1 and 2.
    """

    def __init__(self, low: int = 50, high: int = 150) -> None:
        self.low = low
        self.high = high
        # Instantiate your existing filters to avoid rewriting code!
        self.gaussian = GaussianFilter()
        self.sobel = SobelFilter()

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Step 1: Noise Reduction
        # We use your GaussianFilter to blur the image.
        # (It automatically handles grayscale conversion inside _convolve)
        smoothed = self.gaussian.apply(image)

        # Step 2: Gradient Calculation
        # We use your SobelFilter to get the X and Y gradients
        gx, gy = self.sobel.apply_xy(smoothed)

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
    Detects straight lines using a from-scratch Standard Hough Transform 
    and superimposes them on the original image.
    """

    def __init__(
        self,
        rho_res: float = 1.0,
        theta_res: float = np.pi / 180.0,
        threshold: int = 80,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        self._canny     = CannyFilter(canny_low, canny_high)
        self.rho_res    = rho_res
        self.theta_res  = theta_res
        self.threshold  = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        edges = self._canny.apply(image)
        output = to_bgr(image)
        
        h, w = edges.shape
        diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
        
        # Define parameter spaces
        rhos = np.arange(-diag_len, diag_len + 1, self.rho_res)
        thetas = np.arange(0, np.pi, self.theta_res)
        
        # Accumulator array
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
        y_idxs, x_idxs = np.nonzero(edges)
        
        # Vectorized voting
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        for i in range(len(x_idxs)):
            x, y = x_idxs[i], y_idxs[i]
            # Calculate rho for all thetas simultaneously
            rho_vals = x * cos_t + y * sin_t
            # Find closest indices in the rhos array
            rho_idxs = np.round((rho_vals + diag_len) / self.rho_res).astype(int)
            
            # Cast votes
            for t_idx, r_idx in enumerate(rho_idxs):
                accumulator[r_idx, t_idx] += 1

        # Extract local maxima (peaks) based on threshold
        peaks = np.argwhere(accumulator >= self.threshold)
        
        # Convert polar parameters back to Cartesian lines for drawing
        for r_idx, t_idx in peaks:
            rho = rhos[r_idx]
            theta = thetas[t_idx]
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Extend lines across the image
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        return output

# 5. Hough circle detector

class HoughCirclesFilter(Base):
    """
    Detects circles using a from-scratch Hough Transform by voting 
    in a 3D parameter space (x, y, radius).
    """

    def __init__(
        self,
        min_radius: int = 10,
        max_radius: int = 50,
        threshold: int = 100,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        self._canny = CannyFilter(canny_low, canny_high)
        self.min_radius = max(1, min_radius)
        self.max_radius = max(self.min_radius + 1, max_radius)
        self.threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        edges = self._canny.apply(image)
        output = to_bgr(image)
        
        h, w = edges.shape
        y_idxs, x_idxs = np.nonzero(edges)
        
        # Pre-compute angles to speed up voting (5-degree steps for efficiency)
        thetas = np.deg2rad(np.arange(0, 360, 5))
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        # Iterate over possible radii to save 3D matrix memory
        for r in range(self.min_radius, self.max_radius):
            acc = np.zeros((h, w), dtype=np.int32)
            
            for x, y in zip(x_idxs, y_idxs):
                # Calculate candidate centers
                a_vals = np.round(x - r * cos_t).astype(int)
                b_vals = np.round(y - r * sin_t).astype(int)
                
                # Keep only centers that fall within image boundaries
                valid_mask = (a_vals >= 0) & (a_vals < w) & (b_vals >= 0) & (b_vals < h)
                a_valid = a_vals[valid_mask]
                b_valid = b_vals[valid_mask]
                
                # Accumulate votes
                np.add.at(acc, (b_valid, a_valid), 1)
                
            # Find peaks for this specific radius
            peaks = np.argwhere(acc >= self.threshold)
            for cy, cx in peaks:
                cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(output, (cx, cy), 2, (0, 0, 255), 3) # Center dot
                
        return output


# 6. Hough ellipse detector

class HoughEllipsesFilter(Base):
    """
    Fits ellipses to contours by calculating the spatial moments and 
    covariance matrix (PCA) from scratch, replacing cv2.fitEllipse.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_points: int = 10,
        min_area: float = 120.0,
        max_area_ratio: float = 0.9,
    ) -> None:
        self._canny = CannyFilter(canny_low, canny_high)
        self.min_points = min_points
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def _fit_ellipse_pca(self, points: np.ndarray):
        """Fits an ellipse to a set of 2D points using covariance."""
        pts = points[:, 0, :].astype(np.float64)
        n = len(pts)
        if n < 5:
            return None
            
        x, y = pts[:, 0], pts[:, 1]
        x_mean, y_mean = np.mean(x), np.mean(y)
        
        # Calculate covariance matrix components
        u11 = np.sum((x - x_mean)**2) / n
        u22 = np.sum((y - y_mean)**2) / n
        u12 = np.sum((x - x_mean) * (y - y_mean)) / n
        
        # Find eigenvalues to get axis lengths
        trace = u11 + u22
        det = u11 * u22 - u12**2
        sqrt_term = np.sqrt(max(0, trace**2 - 4 * det))
        
        lambda1 = (trace + sqrt_term) / 2
        lambda2 = (trace - sqrt_term) / 2
        
        # Find orientation angle
        if u12 != 0:
            angle = 0.5 * np.arctan2(2 * u12, u11 - u22)
        else:
            angle = 0 if u11 > u22 else np.pi / 2
            
        angle_deg = np.rad2deg(angle)
        
        # Scale eigenvalues to approximate bounding axes (empirical scaling for boundaries)
        major_axis = 2.0 * np.sqrt(lambda1) * 2.82 
        minor_axis = 2.0 * np.sqrt(lambda2) * 2.82
        
        return ((x_mean, y_mean), (major_axis, minor_axis), angle_deg)

    def apply(self, image: np.ndarray) -> np.ndarray:
        edges = self._canny.apply(image)
        output = to_bgr(image)

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        h, w = edges.shape
        max_area = float(h * w) * self.max_area_ratio
        
        # We extract structural boundaries, then mathematically fit them ourselves
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for cnt in contours:
            if len(cnt) < max(self.min_points, 5):
                continue

            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > max_area:
                continue

            hull = cv2.convexHull(cnt)
            
            # Apply our custom mathematical fit instead of cv2.fitEllipse
            ellipse = self._fit_ellipse_pca(hull)
            if ellipse is None:
                continue
                
            (center, axes, angle) = ellipse
            major, minor = max(axes), min(axes)
            
            if minor <= 0 or (major / minor) > 8.0:
                continue

            # Draw the resulting math model back onto the image
            cv2.ellipse(
                output, 
                (int(center[0]), int(center[1])), 
                (int(major / 2), int(minor / 2)), 
                angle, 0, 360, (255, 0, 0), 2
            )
            
        return output

# 7. Active Contour (Snake)

class ActiveContourFilter(Base):
    """
    Greedy Active Contour (snake) implementation with Local Normalization.
    
    The snake evolves by minimizing a combined energy functional:
    - Continuity energy: keeps points evenly spaced
    - Curvature energy: keeps the contour smooth
    - External energy: attracts the contour toward image edges
    """

    def __init__(
        self,
        n_points: int = 200,
        alpha: float = 0.01,
        beta: float = 0.1,
        gamma: float = 0.01,
        iterations: int = 200,
        w_size: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        n_points : int
            Number of control points on the contour.
        alpha : float
            Elasticity weight (continuity energy).
        beta : float
            Stiffness weight (curvature energy).
        gamma : float
            External energy weight (edge attraction).
        iterations : int
            Number of evolution iterations.
        w_size : int
            Search window radius for greedy optimization.
        """
        self.n_points = n_points
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iterations = iterations
        self.w_size = w_size
        
        # Reuse our filter implementations
        self._gaussian = GaussianFilter()
        self._canny = CannyFilter(low=30, high=100)
        
        # Results populated after apply()
        self.initial_contour: np.ndarray | None = None
        self.contour: np.ndarray | None = None
        self.chain_code: list[int] = []
        self.perimeter: float = 0.0
        self.area: float = 0.0

    def _init_contour_from_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Initialize contour by detecting object boundary via thresholding,
        then expanding the convex hull slightly outward.
        """
        blurred = self._gaussian.apply(gray)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Check if background is dominant at corners; if so, invert
        h, w = gray.shape
        corners = [thresh[0, 0], thresh[0, w - 1], thresh[h - 1, 0], thresh[h - 1, w - 1]]
        if sum(int(c) for c in corners) > 255 * 2:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Fallback: ellipse in center if no contours found
        if len(contours) == 0:
            return self._create_ellipse_contour(w // 2, h // 2, min(h, w) // 2.5)

        # Use convex hull of largest contour, expanded slightly
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        hull_points = hull[:, 0, :].astype(np.float64)

        # Expand hull outward from centroid
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            center = np.array([cx, cy])
            hull_points = center + (hull_points - center) * 1.05

        return self._resample_contour(hull_points)

    def _create_ellipse_contour(
        self, cx: float, cy: float, radius: float
    ) -> np.ndarray:
        """Create an elliptical contour centered at (cx, cy)."""
        t = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        return np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)])

    def _resample_contour(self, points: np.ndarray) -> np.ndarray:
        """Resample a polygon to have exactly n_points evenly spaced."""
        # Close the polygon
        closed = np.vstack((points, points[0]))
        
        # Compute cumulative arc length
        diffs = np.diff(closed, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(distances)))

        # Interpolate to get evenly spaced points
        target_dist = np.linspace(0, cum_dist[-1], self.n_points, endpoint=False)
        x_interp = np.interp(target_dist, cum_dist, closed[:, 0])
        y_interp = np.interp(target_dist, cum_dist, closed[:, 1])

        return np.column_stack([x_interp, y_interp])

    def _compute_external_energy(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute external energy using distance transform.
        
        Creates a potential field that attracts snake points toward edges
        from a distance. Points farther from edges have higher energy.
        """
        # Detect edges using our Canny implementation
        edges = self._canny.apply(gray)

        # Invert: edges become 0 (black), background becomes 255 (white)
        inverted = cv2.bitwise_not(edges)

        # Distance transform: value at each pixel = distance to nearest edge
        # Snake minimizes energy, so it will move toward edges (where dist=0)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

        # Normalize to [0, 1]
        if dist.max() > 0:
            dist = dist / dist.max()

        return dist

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range (local normalization)."""
        min_val, max_val = arr.min(), arr.max()
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

    def _evolve_snake(
        self, snake: np.ndarray, energy: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """
        Evolve the snake for one iteration using greedy optimization.
        
        For each point, search within a local window and pick the position
        that minimizes the total energy (continuity + curvature + external).
        """
        n = len(snake)
        r = self.w_size

        # Compute mean distance for continuity energy
        diffs = np.diff(np.vstack((snake, snake[0])), axis=0)
        d_mean = np.mean(np.linalg.norm(diffs, axis=1))

        for i in range(n):
            original = snake[i].copy()
            p_prev = snake[(i - 1) % n]
            p_next = snake[(i + 1) % n]

            # Collect candidate positions within search window
            candidates = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx = int(original[0] + dx)
                    ny = int(original[1] + dy)
                    
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue

                    curr = np.array([nx, ny], dtype=np.float64)

                    # Internal energies
                    e_cont = (d_mean - np.linalg.norm(curr - p_prev)) ** 2
                    e_curv = np.linalg.norm(p_next - 2 * curr + p_prev) ** 2
                    
                    # External energy from distance transform
                    e_ext = energy[ny, nx]

                    candidates.append([nx, ny, e_cont, e_curv, e_ext])

            if not candidates:
                continue

            candidates = np.array(candidates)

            # Local normalization of each energy term
            e_cont_norm = self._normalize(candidates[:, 2])
            e_curv_norm = self._normalize(candidates[:, 3])
            e_ext_norm = self._normalize(candidates[:, 4])

            # Weighted sum of normalized energies
            total_energy = (
                self.alpha * e_cont_norm
                + self.beta * e_curv_norm
                + self.gamma * e_ext_norm
            )

            # Move to position with minimum total energy
            best_idx = np.argmin(total_energy)
            snake[i] = candidates[best_idx, :2]

        return snake

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the active contour algorithm.
        
        Returns
        -------
        np.ndarray
            Original image with the evolved contour drawn in green.
        """
        gray = to_gray(image)
        h, w = gray.shape
        
        # Compute external energy field
        energy = self._compute_external_energy(gray)

        # Initialize contour
        snake = self._init_contour_from_edges(gray)
        self.initial_contour = snake.astype(np.int32).reshape(-1, 1, 2)

        # Evolve the snake
        for _ in range(self.iterations):
            snake = self._evolve_snake(snake, energy, h, w)

        # Store results
        self.contour = snake.astype(np.int32).reshape(-1, 1, 2)
        self.chain_code = contour_to_chain_code(self.contour)
        self.perimeter = chain_code_perimeter(self.chain_code)
        self.area = contour_area(self.contour)

        # Draw contour on output image
        output = to_bgr(image)
        cv2.polylines(
            output, [self.contour], isClosed=True, color=(0, 255, 0), thickness=2
        )
        return output