"""
Track entity — single vehicle track with alpha-beta smoothed speed estimation.
SpeedEstimator is tightly coupled to Track and lives here for locality.
"""

import cv2
import numpy as np
from collections import deque


class SpeedEstimator:
    """
    Stable real-time speed estimator for a single track.

    Pipeline per frame:
      raw centroid → alpha-beta filter → rolling buffer (≤20 pts)
      → windowed weighted polyfit → CameraModel conversion (px/s → km/h)
      → median spike rejection → EWMA output smoothing
    """

    WINDOW_S  = 0.9    # regression window (seconds)
    AB_ALPHA  = 0.35   # position correction gain
    AB_BETA   = 0.15   # velocity correction gain
    JUMP_THR  = 80     # px jump — reduce gain above this
    EWMA_A    = 0.25   # output smoothing (lower → smoother, laggier)
    SPIKE_K   = 2.8    # reject raw estimate if > K × recent_median
    MIN_PTS   = 3      # minimum buffer points for fit
    MIN_DT    = 0.06   # minimum time span in window (s)
    MAX_KMH   = 200    # hard physical cap

    def __init__(self, fps: float):
        self.fps        = fps
        self._buf       = deque(maxlen=20)   # (fid, t, sx, sy)
        self._speed_buf = deque(maxlen=8)    # recent raw km/h (spike window)
        self.speed_kmh  = 0.0
        self._sx = self._sy = None
        self._vx = self._vy = 0.0

    def push(self, fid: int, cx: float, cy: float) -> None:
        t = fid / self.fps
        if self._sx is None:
            self._sx, self._sy = cx, cy
            self._buf.append((fid, t, cx, cy))
            return
        px_pred = self._sx + self._vx
        py_pred = self._sy + self._vy
        rx, ry  = cx - px_pred, cy - py_pred
        jump    = np.hypot(rx, ry)
        alpha   = self.AB_ALPHA if jump < self.JUMP_THR else self.AB_ALPHA * 0.25
        beta    = self.AB_BETA  if jump < self.JUMP_THR else 0.0
        self._sx   = px_pred + alpha * rx
        self._sy   = py_pred + alpha * ry
        self._vx  += beta * rx
        self._vy  += beta * ry
        self._buf.append((fid, t, self._sx, self._sy))

    def estimate(self, cam, ref_row: float) -> float:
        if len(self._buf) < self.MIN_PTS:
            return self.speed_kmh
        t_now = self._buf[-1][1]
        pts   = [(t, sx, sy) for _, t, sx, sy in self._buf
                 if t_now - t <= self.WINDOW_S]
        if len(pts) < self.MIN_PTS:
            return self.speed_kmh
        times = np.array([p[0] for p in pts])
        xs    = np.array([p[1] for p in pts])
        ys    = np.array([p[2] for p in pts])
        dt    = times[-1] - times[0]
        if dt < self.MIN_DT:
            return self.speed_kmh
        w    = np.exp(np.linspace(-2.0, 0.0, len(times)))
        vx_s = np.polyfit(times, xs, 1, w=w)[0]
        vy_s = np.polyfit(times, ys, 1, w=w)[0]
        raw  = cam.pixel_speed_to_kmh(vx_s, vy_s, ref_row)
        if not (0.0 < raw < self.MAX_KMH):
            return self.speed_kmh
        if len(self._speed_buf) >= 4:
            med = float(np.median(list(self._speed_buf)))
            if med > 1.0 and raw > med * self.SPIKE_K:
                return self.speed_kmh
        self._speed_buf.append(raw)
        self.speed_kmh = (raw if self.speed_kmh < 0.5
                          else self.EWMA_A * raw + (1.0 - self.EWMA_A) * self.speed_kmh)
        return self.speed_kmh

    def inherit(self, other: "SpeedEstimator") -> None:
        self._buf       = deque(other._buf,       maxlen=other._buf.maxlen)
        self._speed_buf = deque(other._speed_buf, maxlen=other._speed_buf.maxlen)
        self.speed_kmh  = other.speed_kmh
        self._sx, self._sy = other._sx, other._sy
        self._vx, self._vy = other._vx, other._vy

    def reset(self) -> None:
        self._buf.clear(); self._speed_buf.clear()
        self.speed_kmh = 0.0
        self._sx = self._sy = None
        self._vx = self._vy = 0.0


class Track:
    """Single vehicle track — bbox, centroid history, speed, crossing state."""

    _next: int = 1

    def __init__(self, box: tuple, fid: int, fps: float):
        self.id        = Track._next; Track._next += 1
        self.bbox      = box
        self.cx        = box[0] + box[2] / 2.0
        self.cy        = box[1] + box[3] / 2.0
        self.fps       = fps
        self.age       = 0
        self.hits      = 1
        self.speed_kmh = 0.0
        self.speed_samples: list = []
        self.first_fid = fid
        self.last_fid  = fid
        self.history   = deque(maxlen=150)
        self.history.append((fid, self.cx, self.cy))
        self._est      = SpeedEstimator(fps)
        self._est.push(fid, self.cx, self.cy)
        # Line crossing state
        self.prev_cx      = self.cx
        self.prev_cy      = self.cy
        self.line_side    = None
        self.crossed      = False
        self.cross_dir    = None
        self.speed_reported = False   # True once speed locked at crossing moment
        # Unique per-track BGR color
        h = (self.id * 47) % 180
        self.color = tuple(int(c) for c in
                           cv2.cvtColor(np.uint8([[[h, 200, 230]]]),
                                        cv2.COLOR_HSV2BGR)[0][0])

    def predict(self) -> tuple:
        if len(self.history) < 2:
            return self.cx, self.cy
        vx = self.history[-1][1] - self.history[-2][1]
        vy = self.history[-1][2] - self.history[-2][2]
        return self.history[-1][1] + vx, self.history[-1][2] + vy

    def update(self, box: tuple, fid: int) -> None:
        self.prev_cx = self.cx
        self.prev_cy = self.cy
        self.bbox    = box
        self.cx      = box[0] + box[2] / 2.0
        self.cy      = box[1] + box[3] / 2.0
        self.age     = 0
        self.hits   += 1
        self.last_fid = fid
        self.history.append((fid, self.cx, self.cy))
        # Feed estimator every frame so crossing speed works even without an ROI.
        self._est.push(fid, self.cx, self.cy)

    def calc_speed(self, cam) -> None:
        # _est.push() is now done in update(); just run the regression here.
        ref_row = self.cy + self.bbox[3] / 2.0
        kmh = self._est.estimate(cam, ref_row)
        self.speed_kmh = kmh
        if kmh > 0:
            self.speed_samples.append(kmh)

    def lock_crossing_speed(self, cam) -> float:
        """Compute and freeze speed at the moment of line crossing.

        Called once per track when it crosses the counting line.
        Uses the alpha-beta filtered centroid history already in the estimator.
        Returns km/h; also updates self.speed_kmh and sets self.speed_reported.
        """
        if self.speed_reported:
            return self.speed_kmh
        if cam is None:
            import warnings
            warnings.warn(f"Track #{self.id}: no CameraModel — crossing speed unavailable",
                          stacklevel=2)
            self.speed_reported = True
            return self.speed_kmh
        ref_row = self.cy + self.bbox[3] / 2.0
        kmh = self._est.estimate(cam, ref_row)
        # Require minimal displacement: at least MIN_PTS frames of data
        if kmh > 0:
            self.speed_kmh = kmh
            if kmh not in self.speed_samples:
                self.speed_samples.append(kmh)
        self.speed_reported = True
        return self.speed_kmh

    @property
    def vel_dir(self) -> tuple:
        """Normalized forward direction (dvx, dvy) in pixel space. (0,0) if stationary."""
        vx, vy = self._est._vx, self._est._vy
        mag = (vx * vx + vy * vy) ** 0.5
        return (vx / mag, vy / mag) if mag > 1e-6 else (0.0, 0.0)

    @property
    def trail(self) -> list:
        return [(int(h[1]), int(h[2])) for h in self.history]

    @property
    def duration(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return (self.history[-1][0] - self.history[0][0]) / self.fps
