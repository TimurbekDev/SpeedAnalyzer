"""ORB-based video stabilizer — compensates camera shake via affine warp."""
import cv2
import numpy as np
from collections import deque


class VideoStabilizer:
    def __init__(self):
        self.enabled        = True
        self.stabilize_view = True
        self.orb  = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_HARRIS_SCORE)
        self.bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_gray = self.prev_kp = self.prev_des = None
        self.shift_history = deque(maxlen=30)
        self.dx = self.dy  = 0.0
        self.confidence    = 0.0
        self.total_frames  = 0
        self.stable_frames = 0

    def reset(self) -> None:
        self.prev_gray = self.prev_kp = self.prev_des = None
        self.shift_history.clear()
        self.dx = self.dy = 0.0
        self.total_frames = self.stable_frames = 0

    def process(self, frame: np.ndarray) -> tuple:
        """Returns (stabilized_frame, dx, dy, confidence)."""
        self.total_frames += 1
        if not self.enabled:
            return frame, 0.0, 0.0, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None or des is None or self.prev_des is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return frame, 0.0, 0.0, 0.0

        try:
            matches = self.bf.knnMatch(self.prev_des, des, k=2)
        except Exception:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return frame, 0.0, 0.0, 0.0

        good = [m for pair in matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance]
        self.confidence = len(good) / max(len(matches), 1)

        if len(good) < 10:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            self.dx = self.dy = 0.0
            return frame, 0.0, 0.0, 0.0

        src = np.float32([self.prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt          for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
        if M is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            self.dx = self.dy = 0.0
            return frame, 0.0, 0.0, 0.0

        self.dx, self.dy = float(M[0, 2]), float(M[1, 2])
        self.shift_history.append((self.dx, self.dy))
        if np.hypot(self.dx, self.dy) < 1.0:
            self.stable_frames += 1

        result = frame
        if self.stabilize_view and np.hypot(self.dx, self.dy) > 0.5:
            h, w = frame.shape[:2]
            n = len(self.shift_history)
            sx = np.mean([s[0] for s in self.shift_history]) if n >= 3 else self.dx
            sy = np.mean([s[1] for s in self.shift_history]) if n >= 3 else self.dy
            result = cv2.warpAffine(frame,
                                    np.float32([[1, 0, -sx], [0, 1, -sy]]),
                                    (w, h), borderMode=cv2.BORDER_REPLICATE)

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return result, self.dx, self.dy, self.confidence

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return "OFF"
        pct = (self.stable_frames / max(self.total_frames, 1)) * 100
        return f"dx={self.dx:.1f} dy={self.dy:.1f} ({pct:.0f}% stable)"
