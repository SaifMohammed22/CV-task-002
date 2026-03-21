from __future__ import annotations

import numpy as np
import cv2

from .base import Base
from src.utils import to_gray, to_bgr, ensure_uint8



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

# 7. Active Contour (Snake)

# 7. Active Contour (Snake)

class ActiveContourFilter(Base):
    """
    Greedy Active Contour (snake) implementation with Local Normalization.
    """

    def __init__(
            self,
            n_points: int = 200,
            alpha: float = 0.01,  # elasticity  (internal – continuity)
            beta: float = 0.1,  # stiffness   (internal – curvature)
            gamma: float = 0.01,  # step size   (external - edge attraction)
            iterations: int = 200,
            w_size: int = 3,
    ) -> None:
        self.n_points = n_points
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iterations = iterations
        self.w_size = w_size

    @staticmethod
    def _init_contour_from_edges(gray: np.ndarray, n: int) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = gray.shape
        corners_sum = int(thresh[0, 0]) + int(thresh[0, w - 1]) + int(thresh[h - 1, 0]) + int(thresh[h - 1, w - 1])
        if corners_sum > 255 * 2:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            cx, cy = w // 2, h // 2
            r = min(h, w) // 2.5
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]).astype(np.float64)

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        hull_points = hull[:, 0, :]

        # تقليل نسبة التكبير المبدئي إلى 5% فقط (بدلاً من 15%) ليكون أقرب للشكل
        M = cv2.moments(largest)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = np.array([cx, cy])
            hull_points = center + (hull_points - center) * 1.05

        hull_points = np.vstack((hull_points, hull_points[0]))

        diffs = np.diff(hull_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(distances)))

        target_dist = np.linspace(0, cum_dist[-1], n, endpoint=False)
        x_interp = np.interp(target_dist, cum_dist, hull_points[:, 0])
        y_interp = np.interp(target_dist, cum_dist, hull_points[:, 1])

        return np.column_stack([x_interp, y_interp]).astype(np.float64)

    @staticmethod
    def _external_energy(gray: np.ndarray) -> np.ndarray:
        """استخدام Distance Transform لصنع مجال مغناطيسي يجذب النقاط للحواف من مسافة بعيدة"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # استخراج الحواف
        edges = cv2.Canny(blurred, 30, 100)

        # عكس الألوان (نجعل الحواف سوداء والخلفية بيضاء)
        inverted = cv2.bitwise_not(edges)

        # حساب المسافة من كل بيكسل إلى أقرب حافة
        # هذا سيجعل الـ Snake يتدحرج تلقائياً نحو الحافة (لأنه يبحث عن أقل طاقة)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

        if dist.max() > 0:
            dist = dist / dist.max()

        return dist  # نرجعها بالموجب لأننا نريد للـ Snake أن يقلل المسافة (الوصول لـ 0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = to_gray(image)
        h, w = gray.shape
        energy = self._external_energy(gray)

        snake = self._init_contour_from_edges(gray, self.n_points)
        r = self.w_size

        for _ in range(self.iterations):
            # حساب متوسط المسافة لخطوة Continuity مرة واحدة كل لفة
            diffs = np.diff(np.vstack((snake, snake[0])), axis=0)
            d_mean = np.mean(np.linalg.norm(diffs, axis=1))

            for i in range(len(snake)):
                original = snake[i].copy()
                n = len(snake)
                p_prev = snake[(i - 1) % n]
                p_next = snake[(i + 1) % n]

                candidates = []
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx = int(original[0] + dx)
                        ny = int(original[1] + dy)
                        if not (0 <= nx < w and 0 <= ny < h):
                            continue

                        curr_point = np.array([nx, ny], dtype=np.float64)

                        # حساب الطاقات (بدون ضربها في المعاملات الآن)
                        e_cont = (d_mean - np.linalg.norm(curr_point - p_prev)) ** 2
                        e_curv = np.linalg.norm(p_next - 2 * curr_point + p_prev) ** 2
                        e_ext = energy[ny, nx]

                        candidates.append([nx, ny, e_cont, e_curv, e_ext])

                if not candidates:
                    continue

                candidates = np.array(candidates)

                # تطبيع الطاقات محلياً (Local Normalization) - مهم جداً لنجاح الخوارزمية
                def normalize(arr):
                    m, M = np.min(arr), np.max(arr)
                    return (arr - m) / (M - m) if M > m else np.zeros_like(arr)

                e_cont_norm = normalize(candidates[:, 2])
                e_curv_norm = normalize(candidates[:, 3])
                e_ext_norm = normalize(candidates[:, 4])

                # حساب الطاقة الكلية بعد التطبيع
                total_e = (self.alpha * e_cont_norm +
                           self.beta * e_curv_norm +
                           self.gamma * e_ext_norm)

                best_idx = np.argmin(total_e)
                snake[i] = [candidates[best_idx, 0], candidates[best_idx, 1]]

        self.contour = snake.astype(np.int32).reshape(-1, 1, 2)
        from src.utils import contour_to_chain_code, chain_code_perimeter, contour_area
        self.chain_code = contour_to_chain_code(self.contour)
        self.perimeter = chain_code_perimeter(self.chain_code)
        self.area = contour_area(self.contour)

        output = to_bgr(image)
        cv2.polylines(output, [self.contour], isClosed=True, color=(0, 255, 0), thickness=2)
        return output