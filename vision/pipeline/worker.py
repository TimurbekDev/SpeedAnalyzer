"""
Worker thread — consumes (fid, frame) from in_q, runs detect+track,
pushes (frame, tracks, finished, crossings, following) to out_q.

Thread-safe cross_line and proximity zone updates via Lock.
Speed/distance processing gated by threading.Events set from the UI thread.
"""
import queue
import threading
import numpy as np
from config.defaults import FRAME_SKIP


class Worker(threading.Thread):
    def __init__(self, yolo_det, bg_det, enhancer, tracker, in_q, out_q,
                 cross_line=None,
                 prox_zone=None, prox_calc=None,
                 prox_analytics=None, prox_area_m2=5000.0,
                 following_calc=None, crossing_calc=None,
                 speed_event:   threading.Event | None = None,
                 dist_event:    threading.Event | None = None):
        super().__init__(daemon=True)
        self.yolo  = yolo_det
        self.bg    = bg_det
        self.enh   = enhancer
        self.trk   = tracker
        self.in_q  = in_q
        self.out_q = out_q
        self._stop = threading.Event()
        self._fid  = 0
        self._last_boxes: list              = []
        self._cross_line                    = cross_line
        self._prox_zone:  np.ndarray | None = prox_zone
        self._prox_calc                     = prox_calc
        self._prox_analytics                = prox_analytics
        self._prox_area_m2: float           = prox_area_m2
        self._following_calc                = following_calc
        self._crossing_calc                 = crossing_calc
        self._lock = threading.Lock()

        # Feature gates — checked each frame without acquiring the main lock.
        # Use threading.Event so the UI thread can toggle them safely at any time.
        self._speed_en = speed_event or _AlwaysSet()
        self._dist_en  = dist_event  or _AlwaysSet()

    # ── Thread-safe setters ───────────────────────────────────────────────────

    def set_cross_line(self, cl) -> None:
        with self._lock:
            self._cross_line = cl

    def set_prox_zone(self, mask: np.ndarray | None,
                      area_m2: float = 5000.0) -> None:
        with self._lock:
            self._prox_zone     = mask
            self._prox_area_m2  = area_m2

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _in_roi(cx: float, cy: float, mask) -> bool:
        if mask is None:
            return False
        xi, yi = int(cx), int(cy)
        h, w = mask.shape[:2]
        return 0 <= yi < h and 0 <= xi < w and mask[yi, xi] > 0

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                self.out_q.put(None)
                break

            fid, frame = item
            enhanced   = self.enh.process(frame)

            with self._lock:
                cl   = self._cross_line
                pz   = self._prox_zone
                pc   = self._prox_calc
                pa   = self._prox_analytics
                pa2  = self._prox_area_m2

            if self._fid % max(1, FRAME_SKIP) == 0:
                if self.yolo and self.yolo.model:
                    self._last_boxes = self.yolo.detect(enhanced)
                else:
                    self._last_boxes = self.bg.detect(enhanced)
            self._fid += 1

            tracks   = self.trk.update(self._last_boxes, fid)
            finished = self.trk.finished[:]
            self.trk.finished.clear()

            cam = self.trk.cam

            # ── Speed calculation (gated) ──────────────────────────────────
            if self._speed_en.is_set():
                for t in tracks:
                    t.calc_speed(cam)

            # ── Crossing detection (always active) ─────────────────────────
            crossings: list[dict] = []
            if cl and cl.active:
                for t in self.trk.tracks:
                    ev = cl.check_track(t, fid, cam=cam)
                    if ev:
                        # Attach following distance only when dist module is on.
                        if (self._dist_en.is_set()
                                and self._crossing_calc is not None):
                            ev["crossing_dist"] = self._crossing_calc.find_rear(
                                ev, self.trk.tracks, cl)
                        crossings.append(ev)

            # ── Proximity analytics (gated) ────────────────────────────────
            if self._dist_en.is_set() and pz is not None \
                    and pc is not None and pa is not None:
                zone_tracks = [t for t in self.trk.tracks
                               if self._in_roi(t.cx, t.cy, pz)]
                pairs = pc.compute(zone_tracks)
                pa.update(pairs, len(zone_tracks), pa2)

            # ── Following distance (gated) ─────────────────────────────────
            following = (self._following_calc.compute(tracks)
                         if self._dist_en.is_set()
                         and self._following_calc is not None else [])

            result = (enhanced, tracks, finished, crossings, following)
            try:
                self.out_q.put_nowait(result)
            except queue.Full:
                try:    self.out_q.get_nowait()
                except queue.Empty: pass
                try:    self.out_q.put_nowait(result)
                except queue.Full:  pass

    def stop(self) -> None:
        self._stop.set()


class _AlwaysSet:
    """Drop-in for threading.Event that is always set (feature always enabled)."""
    def is_set(self) -> bool:
        return True
