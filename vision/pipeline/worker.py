"""
Worker thread — consumes (fid, frame) from in_q, runs detect+track,
pushes (frame, tracks, finished, crossings) to out_q.

Thread-safe cross_line, roi_mask, and proximity zone updates via Lock.
"""
import queue
import threading
import numpy as np
from config.defaults import FRAME_SKIP


class Worker(threading.Thread):
    def __init__(self, yolo_det, bg_det, enhancer, tracker, in_q, out_q,
                 cross_line=None,
                 prox_zone=None, prox_calc=None,
                 prox_analytics=None, prox_area_m2=5000.0):
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
        self._roi_mask:   np.ndarray | None = None
        self._cross_line                    = cross_line
        self._prox_zone:  np.ndarray | None = prox_zone
        self._prox_calc                     = prox_calc
        self._prox_analytics                = prox_analytics
        self._prox_area_m2: float           = prox_area_m2
        self._lock = threading.Lock()

    # ── Thread-safe setters ───────────────────────────────────────────────────

    def set_roi(self, mask: np.ndarray) -> None:
        with self._lock:
            self._roi_mask = mask

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
                roi  = self._roi_mask
                cl   = self._cross_line
                pz   = self._prox_zone
                pc   = self._prox_calc
                pa   = self._prox_analytics
                pa2  = self._prox_area_m2

            if self._fid % max(1, FRAME_SKIP) == 0:
                if self.yolo and self.yolo.model:
                    self._last_boxes = self.yolo.detect(enhanced, None)
                else:
                    self._last_boxes = self.bg.detect(enhanced, roi)
            self._fid += 1

            tracks   = self.trk.update(self._last_boxes, fid)
            finished = self.trk.finished[:]
            self.trk.finished.clear()

            cam = self.trk.cam
            for t in tracks:
                if self._in_roi(t.cx, t.cy, roi):
                    t.calc_speed(cam)

            # ── Crossing detection ─────────────────────────────────────────
            crossings: list[dict] = []
            if cl and cl.active:
                for t in self.trk.tracks:
                    ev = cl.check_track(t, fid, cam=cam)
                    if ev:
                        crossings.append(ev)

            # ── Proximity analytics ────────────────────────────────────────
            if pz is not None and pc is not None and pa is not None:
                zone_tracks = [t for t in self.trk.tracks
                               if self._in_roi(t.cx, t.cy, pz)]
                pairs = pc.compute(zone_tracks)
                pa.update(pairs, len(zone_tracks), pa2)

            result = (enhanced, tracks, finished, crossings)
            try:
                self.out_q.put_nowait(result)
            except queue.Full:
                try:    self.out_q.get_nowait()
                except queue.Empty: pass
                try:    self.out_q.put_nowait(result)
                except queue.Full:  pass

    def stop(self) -> None:
        self._stop.set()
