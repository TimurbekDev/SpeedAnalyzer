"""
MetricSpeedTracker — per-vehicle speed from ground-plane displacement.

The lane homography (LaneCalibration) already converts a vehicle's ground point
to real metres.  Speed is therefore just metres travelled / seconds elapsed:

    v (km/h) = |P_now − P_ref|_metres / Δt_seconds × 3.6

where P_ref is the oldest sample still inside a short time window.  Using metric
positions (not pixels) makes the speed perspective-correct — a vehicle far down
the lane and one close to the camera are measured on the same real-world scale.

A short window + EWMA smoothing rejects bbox jitter; a hard cap rejects spikes.
"""

from __future__ import annotations

from collections import deque
from math import hypot

from config.defaults import (SPEED_WINDOW_S, SPEED_EWMA,
                             SPEED_MIN_DT, SPEED_MAX_KMH)


class MetricSpeedTracker:
    def __init__(self,
                 window_s: float = SPEED_WINDOW_S,
                 ewma:     float = SPEED_EWMA,
                 min_dt:   float = SPEED_MIN_DT,
                 max_kmh:  float = SPEED_MAX_KMH):
        self.window_s = window_s
        self.ewma     = ewma
        self.min_dt   = min_dt
        self.max_kmh  = max_kmh
        self._hist: dict = {}   # id -> deque[(t, x_m, y_m)]
        self._spd:  dict = {}   # id -> smoothed km/h
        self._vel:  dict = {}   # id -> (vx, vy) metric m/s (travel direction)

    def update(self, tid: int, x_m: float, y_m: float, t: float) -> float:
        buf = self._hist.setdefault(tid, deque(maxlen=64))
        buf.append((t, x_m, y_m))

        # Drop samples older than the window (keep at least two points).
        while len(buf) > 2 and (t - buf[0][0]) > self.window_s:
            buf.popleft()

        t_ref, x_ref, y_ref = buf[0]
        dt = t - t_ref
        if dt >= self.min_dt:
            self._vel[tid] = ((x_m - x_ref) / dt, (y_m - y_ref) / dt)
            raw = hypot(x_m - x_ref, y_m - y_ref) / dt * 3.6   # m/s → km/h
            if 0.0 <= raw <= self.max_kmh:
                prev = self._spd.get(tid)
                self._spd[tid] = (raw if prev is None
                                  else self.ewma * raw + (1.0 - self.ewma) * prev)
        return self._spd.get(tid, 0.0)

    def velocity(self, tid: int):
        """Metric travel direction (vx, vy) in m/s, or None if not yet known."""
        return self._vel.get(tid)

    def keep_only(self, active_ids) -> None:
        """Drop history for vehicles no longer in the ROI (bounds memory)."""
        active = set(active_ids)
        for tid in [k for k in self._hist if k not in active]:
            self._hist.pop(tid, None)
            self._spd.pop(tid, None)
            self._vel.pop(tid, None)

    def reset(self) -> None:
        self._hist.clear()
        self._spd.clear()
        self._vel.clear()
