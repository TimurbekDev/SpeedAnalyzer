"""
Worker thread — ByteTrack multi-lane distance pipeline.

    frame → detect+track ONCE on the whole frame (ByteTrack) → for EACH lane ROI:
    keep the vehicles whose ground point is inside it → per-vehicle speed (metric)
    + gap to the vehicle ahead → log each vehicle ONCE per lane → push to the UI.

Why one track() call, many lanes
─────────────────────────────────
ByteTrack's frame-to-frame state lives inside the YOLO model.  Tracking the
whole frame a single time per frame keeps that state consistent and gives every
lane the SAME stable ids.  (Running one tracker per ROI — calling track() several
times on the same model — corrupts that state and is what made vehicles lose
their id and get logged twice.)  Lanes only *filter* the shared detections.

Every frame handed to the worker is tracked (no frame-skip) so ByteTrack's motion
model stays consistent.
"""

import queue
import threading

from config.defaults import MEASURE_SPEED_GRACE_S
from core.entities.vehicle import Vehicle
from vision.pipeline.speed_tracker import MetricSpeedTracker


class _LaneState:
    """Per-lane measurement state (counts, recorded ids, speed history)."""

    def __init__(self, lane):
        self.lane     = lane            # vision.roi.Lane (roi + calib + name + color)
        self.counted  = set()           # unique ids that entered this lane
        self.recorded = set()           # ids already written to this lane's log
        self.seen_fid = {}              # id -> first fid inside this lane
        self.speed    = MetricSpeedTracker()


class Worker(threading.Thread):
    def __init__(self, detector, in_q, out_q, lanes, fps: float):
        super().__init__(daemon=True)
        self.det    = detector
        self.in_q   = in_q
        self.out_q  = out_q
        self.fps    = max(1.0, float(fps))
        self._states = [_LaneState(l) for l in lanes]
        self._stop   = threading.Event()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _vtime(self, fid: int) -> str:
        secs = int(fid / self.fps)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    @staticmethod
    def _compute_nexts(inside: list, metric: dict, speed) -> dict:
        """
        For each in-lane vehicle, find the vehicle AHEAD of it and the metric gap.
        "Ahead" = the half-plane the vehicle is travelling toward (its metric
        velocity); the closest vehicle there is its follow target.  Falls back to
        nearest neighbour when direction is unknown (just appeared / stopped).
        Lead vehicle → (None, None).  Returns {id: (next_id, gap_m or None)}.
        """
        nexts: dict = {}
        ids = [v.id for v in inside if v.id in metric]
        for tid in ids:
            px, py = metric[tid]
            vel = speed.velocity(tid)
            has_dir = vel is not None and (vel[0] ** 2 + vel[1] ** 2) > 1e-6
            best_id, best_gap = None, float("inf")
            for oid in ids:
                if oid == tid:
                    continue
                ox, oy = metric[oid]
                dx, dy = ox - px, oy - py
                if has_dir and (dx * vel[0] + dy * vel[1]) <= 0.0:
                    continue                      # behind, not ahead
                gap = (dx * dx + dy * dy) ** 0.5
                if gap < best_gap:
                    best_gap, best_id = gap, oid
            nexts[tid] = (best_id, best_gap if best_id is not None else None)
        return nexts

    def _process_lane(self, st: _LaneState, vehicles: list, fid: int, t_sec: float) -> dict:
        """Filter the shared detections to one lane and produce its frame result."""
        roi, calib = st.lane.roi, st.lane.calib
        inside = [v for v in vehicles if roi.contains(*v.ground)]
        inside.sort(key=lambda v: v.id)
        inside_ids = [v.id for v in inside]

        speeds: dict = {}
        metric: dict = {}
        for v in inside:
            st.counted.add(v.id)
            st.seen_fid.setdefault(v.id, fid)
            if calib.ready:
                xm, ym = calib.to_metric(v.ground)
                metric[v.id] = (xm, ym)
                speeds[v.id] = round(st.speed.update(v.id, xm, ym, t_sec), 1)
            else:
                speeds[v.id] = 0.0
        st.speed.keep_only(inside_ids)

        nexts = self._compute_nexts(inside, metric, st.speed)

        records: list = []
        for v in inside:
            if v.id in st.recorded:
                continue
            sp = speeds.get(v.id, 0.0)
            nid, gap = nexts.get(v.id, (None, None))
            waited = (fid - st.seen_fid[v.id]) / self.fps
            if sp > 1 and (gap is not None or waited >= MEASURE_SPEED_GRACE_S):
                st.recorded.add(v.id)
                records.append({
                    "roi":     st.lane.name,
                    "id":      v.id,
                    "speed":   sp,
                    "dist_m":  round(gap, 1) if gap is not None else None,
                    "next_id": nid,
                    "vtime":   self._vtime(fid),
                })

        return {
            "name":       st.lane.name,
            "inside_ids": inside_ids,
            "speeds":     speeds,
            "nexts":      nexts,
            "count":      len(st.counted),
            "records":    records,
        }

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

            # ── Detect + track every vehicle in the frame ONCE (ByteTrack) ──
            vehicles = [Vehicle(tid, (x, y, w, h))
                        for (tid, x, y, w, h) in self.det.track(frame)]
            t_sec = fid / self.fps

            # ── Distribute the shared detections to every lane ──────────────
            lanes_out = [self._process_lane(st, vehicles, fid, t_sec)
                         for st in self._states]

            result = (frame, vehicles, lanes_out)
            try:
                self.out_q.put_nowait(result)
            except queue.Full:
                try:    self.out_q.get_nowait()
                except queue.Empty: pass
                try:    self.out_q.put_nowait(result)
                except queue.Full:  pass

    def stop(self) -> None:
        self._stop.set()
