"""
Lane + RoiManager — multiple road-lane ROIs in one session.

A `Lane` bundles everything that is per-region: the polygon (LaneROI), its own
perspective homography (LaneCalibration) and a stable name + colour.  All lanes
share the same real length/width (set once in the UI); the homography still
differs per lane because each lane's corners are different.

`RoiManager` owns the list of lanes, the current selection (the lane being
drawn / edited / inspected) and hands out names + palette colours.  It holds NO
detection/tracking state — tracking is done ONCE per frame on the whole frame
(see Worker) and the result is distributed to each lane, so all lanes share one
stable set of ByteTrack ids.
"""

from __future__ import annotations

from typing import List, Optional

from config.defaults import ROI_PALETTE
from vision.roi.lane_roi import LaneROI
from vision.calibration import LaneCalibration


class Lane:
    def __init__(self, name: str, color: tuple):
        self.name  = name
        self.color = color
        self.roi   = LaneROI()
        self.calib = LaneCalibration()

    @property
    def valid(self) -> bool:
        return self.roi.valid

    @property
    def ready(self) -> bool:
        return self.calib.ready


class RoiManager:
    def __init__(self):
        self.lanes: List[Lane] = []
        self.selected: int = -1
        self._seq: int = 0          # monotonic — names are never reused

    # ── Lane lifecycle ──────────────────────────────────────────────────────────

    def add(self) -> Lane:
        """Create a new lane, make it the current selection, and return it."""
        self._seq += 1
        color = ROI_PALETTE[(self._seq - 1) % len(ROI_PALETTE)]
        lane  = Lane(f"ROI_{self._seq}", color)
        self.lanes.append(lane)
        self.selected = len(self.lanes) - 1
        return lane

    def remove(self, idx: int) -> bool:
        if not (0 <= idx < len(self.lanes)):
            return False
        self.lanes.pop(idx)
        if not self.lanes:
            self.selected = -1
        else:
            self.selected = min(self.selected, len(self.lanes) - 1)
        return True

    def remove_current(self) -> bool:
        return self.remove(self.selected)

    def clear(self) -> None:
        self.lanes = []
        self.selected = -1
        self._seq = 0

    # ── Selection ───────────────────────────────────────────────────────────────

    def select(self, idx: int) -> None:
        if 0 <= idx < len(self.lanes):
            self.selected = idx

    def current(self) -> Optional[Lane]:
        if 0 <= self.selected < len(self.lanes):
            return self.lanes[self.selected]
        return None

    # ── Queries ─────────────────────────────────────────────────────────────────

    def valid_lanes(self) -> List[Lane]:
        """Lanes with a finished polygon — the ones the worker will measure."""
        return [l for l in self.lanes if l.valid]

    def __len__(self) -> int:
        return len(self.lanes)
