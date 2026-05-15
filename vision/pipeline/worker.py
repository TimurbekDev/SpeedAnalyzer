"""
Worker thread — consumes (fid, frame) from in_q, runs detect+track,
pushes (frame, tracks, finished, crossings) to out_q.

Thread-safe cross_line and roi_mask updates via Lock.
"""
import queue
import threading
import numpy as np
from config.defaults import FRAME_SKIP


class Worker(threading.Thread):
    def __init__(self, yolo_det, bg_det, enhancer, tracker, in_q, out_q,
                 cross_line=None):
        super().__init__(daemon=True)
        self.yolo = yolo_det
        self.bg   = bg_det
        self.enh  = enhancer
        self.trk  = tracker
        self.in_q = in_q
        self.out_q = out_q
        self._stop = threading.Event()
        self._fid  = 0
        self._last_boxes: list               = []
        self._roi_mask:   np.ndarray | None  = None
        self._cross_line                     = cross_line
        self._lock = threading.Lock()

    def set_roi(self, mask: np.ndarray) -> None:
        with self._lock:
            self._roi_mask = mask

    def set_cross_line(self, cl) -> None:
        with self._lock:
            self._cross_line = cl

    @staticmethod
    def _in_roi(cx: float, cy: float, mask) -> bool:
        if mask is None:
            return False
        xi, yi = int(cx), int(cy)
        h, w = mask.shape[:2]
        return 0 <= yi < h and 0 <= xi < w and mask[yi, xi] > 0

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
                roi = self._roi_mask
                cl  = self._cross_line

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

            crossings: list[dict] = []
            if cl and cl.active:
                for t in self.trk.tracks:
                    ev = cl.check_track(t, fid, cam=cam)
                    if ev:
                        crossings.append(ev)

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
