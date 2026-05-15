"""
Multi-object tracker — IoU + centroid cost matrix matching.

Ghost guard prevents double-counting when YOLO re-detects a vehicle
immediately after it crossed the line.  SpeedEstimator state is passed
to the next track (ID-switch recovery) via the dead-estimator pool.
"""
import numpy as np
from collections import deque
from core.entities.track import Track
from config.defaults import (
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
    TRACKER_IOU_THR, TRACKER_DIST_THR,
)


class Tracker:
    def __init__(self, fps: float, cam,
                 max_age:  int   = TRACKER_MAX_AGE,
                 min_hits: int   = TRACKER_MIN_HITS,
                 iou_thr:  float = TRACKER_IOU_THR,
                 dist_thr: int   = TRACKER_DIST_THR):
        self.fps      = fps
        self.cam      = cam
        self.max_age  = max_age
        self.min_hits = min_hits
        self.iou_thr  = iou_thr
        self.dist_thr = dist_thr
        self.tracks:   list[Track] = []
        self.finished: list[Track] = []
        self.counted_ids: set      = set()
        self._ghost_buf:  deque    = deque(maxlen=60)
        self._dead_estimators: deque = deque(maxlen=20)

    def reset(self) -> None:
        self.tracks = []; self.finished = []
        Track._next  = 1
        self.counted_ids.clear()
        self._ghost_buf.clear()
        self._dead_estimators.clear()

    # ── Ghost guard ───────────────────────────────────────────────────────────
    def _is_ghost(self, cx: float, cy: float, fid: int,
                  radius: float = 90, max_frames: int = 45) -> bool:
        return any(np.hypot(cx - px, cy - py) < radius and (fid - pf) < max_frames
                   for px, py, pf in self._ghost_buf)

    def _make_track(self, box: tuple, fid: int) -> Track:
        t = Track(box, fid, self.fps)
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        if self._is_ghost(cx, cy, fid):
            t.crossed = True
            return t
        best_dist, best_est = 90.0, None
        for (dcx, dcy, dfid, dest) in self._dead_estimators:
            if fid - dfid > 45:
                continue
            d = np.hypot(cx - dcx, cy - dcy)
            if d < best_dist:
                best_dist, best_est = d, dest
        if best_est is not None:
            t._est.inherit(best_est)
            t.speed_kmh = best_est.speed_kmh
        return t

    # ── Main update ───────────────────────────────────────────────────────────
    def update(self, boxes: list, fid: int) -> list:
        for t in self.tracks:
            t.age += 1

        if boxes and self.tracks:
            nt, nd = len(self.tracks), len(boxes)
            cost   = np.full((nt, nd), 1e9)
            for i, trk in enumerate(self.tracks):
                px, py = trk.predict()
                tx, ty, tw, th = trk.bbox
                for j, (x, y, w, h) in enumerate(boxes):
                    iou  = self._iou((tx, ty, tx+tw, ty+th), (x, y, x+w, y+h))
                    dist = np.hypot(px - (x + w/2), py - (y + h/2))
                    if iou > self.iou_thr or dist < self.dist_thr:
                        cost[i, j] = (1 - iou) * 80 + dist
            mt, md = set(), set()
            for idx in np.argsort(cost, axis=None):
                i, j = divmod(int(idx), nd)
                if cost[i, j] >= 1e8:
                    break
                if i not in mt and j not in md:
                    self.tracks[i].update(boxes[j], fid)
                    mt.add(i); md.add(j)
            for j in range(nd):
                if j not in md:
                    self.tracks.append(self._make_track(boxes[j], fid))
        elif boxes:
            for b in boxes:
                self.tracks.append(self._make_track(b, fid))

        alive = []
        for t in self.tracks:
            if t.crossed:
                self.counted_ids.add(t.id)
                self._ghost_buf.append((t.cx, t.cy, fid))
                if t.hits >= self.min_hits and t.speed_samples:
                    self.finished.append(t)
            elif t.age > self.max_age:
                self._dead_estimators.append((t.cx, t.cy, fid, t._est))
                if t.hits >= self.min_hits and t.speed_samples:
                    self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.hits >= self.min_hits]

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)
