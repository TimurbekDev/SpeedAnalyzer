"""
Per-frame proximity analytics aggregation — thread-safe.

update() is called from the Worker thread every frame.
snapshot property is read from the UI thread via _poll().
Internal lock makes cross-thread access safe.
"""

import threading
import numpy as np
from collections import deque
from typing import List
from vision.proximity.distance_calculator import VehiclePair


class ProximityAnalytics:
    """
    Aggregates inter-vehicle distance stats for one proximity zone.

    Congestion formula:
        smooth_avg = rolling mean of avg_m over SMOOTH_LEN frames
        congestion = clamp(0, 100,  (1 - smooth_avg / (safe_m * 2)) * 100)
        → 0%  when vehicles are 2× safe distance apart
        → 100% when avg gap approaches 0

    Density: vehicles / 1000 m² of zone area.
    """

    SMOOTH_LEN = 30

    def __init__(self,
                 safe_m:   float = 10.0,
                 warn_m:   float =  5.0,
                 danger_m: float =  2.5):
        self.safe_m   = safe_m
        self.warn_m   = warn_m
        self.danger_m = danger_m

        self._lock           = threading.Lock()
        self._pairs:         List[VehiclePair] = []
        self._min_m:         float = 0.0
        self._max_m:         float = 0.0
        self._avg_m:         float = 0.0
        self._congestion:    float = 0.0
        self._density:       float = 0.0
        self._count:         int   = 0
        self._avg_hist               = deque(maxlen=self.SMOOTH_LEN)

    # ── Worker-thread write ───────────────────────────────────────────────────

    def update(self, pairs: List[VehiclePair],
               vehicle_count: int,
               zone_area_m2: float = 5000.0) -> None:
        dists = [p.m_dist for p in pairs]
        avg_m = float(np.mean(dists)) if dists else 0.0

        with self._lock:
            self._pairs = pairs
            self._count = vehicle_count
            self._min_m = float(min(dists)) if dists else 0.0
            self._max_m = float(max(dists)) if dists else 0.0
            self._avg_m = avg_m

            self._avg_hist.append(avg_m)
            smooth = float(np.mean(self._avg_hist))

            normal = self.safe_m * 2.0
            if vehicle_count == 0:
                self._congestion = 0.0
            elif smooth > 0:
                self._congestion = max(0.0, min(100.0,
                    (1.0 - smooth / max(normal, 0.01)) * 100.0))
            else:
                self._congestion = 100.0

            self._density = (vehicle_count / max(zone_area_m2, 1.0)) * 1000.0

    # ── UI-thread read ────────────────────────────────────────────────────────

    @property
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "min_m":      self._min_m,
                "max_m":      self._max_m,
                "avg_m":      self._avg_m,
                "congestion": self._congestion,
                "density":    self._density,
                "count":      self._count,
                "pairs":      list(self._pairs),
            }

    def reset(self) -> None:
        with self._lock:
            self._pairs = []
            self._avg_hist.clear()
            self._min_m = self._max_m = self._avg_m = 0.0
            self._congestion = self._density = 0.0
            self._count = 0
