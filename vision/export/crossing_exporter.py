"""
Real-time CSV exporter for crossing-distance events.

Each crossing event (vehicle crosses counting line) writes one row to a CSV
file immediately.  The file is opened, written, and closed per row so data is
durable even if the process crashes mid-session.

Deduplication by crossed_vehicle_id prevents duplicate rows when the same
vehicle ID appears in multiple queue-drain cycles.
"""

import csv
import os
from datetime import datetime
from typing import Set, Optional

from vision.proximity.crossing_distance import CrossingDistanceResult


class CrossingExporter:
    """
    Appends CrossingDistanceResult rows to a UTF-8 CSV file in real time.

    Usage
    ─────
        exporter = CrossingExporter("crossing_distances.csv")
        exporter.append(result)   # called from main (UI) thread
        exporter.reset()          # call on video reopen / clear-all
    """

    HEADERS = [
        "timestamp",
        "crossed_vehicle_id",
        "rear_vehicle_id",
        "lane_id",
        "distance_m",
        "crossed_speed_kmh",
        "rear_speed_kmh",
    ]

    def __init__(self, path: str = "crossing_distances.csv"):
        self._path: str       = path
        self._seen: Set[int]  = set()
        self._ensure_header()

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_path(self, path: str) -> None:
        """Switch output file (e.g. per-session filename). Writes header if new."""
        self._path = path
        self._ensure_header()

    # ── Write ─────────────────────────────────────────────────────────────────

    def append(self, result: CrossingDistanceResult) -> bool:
        """
        Write one result row.

        Returns True on success, False if this crossed_id was already exported
        (duplicate guard).
        """
        if result.crossed_id in self._seen:
            return False
        self._seen.add(result.crossed_id)

        row = [
            result.timestamp,
            result.crossed_id,
            result.rear_id if result.rear_id is not None else "",
            result.lane_id,
            f"{result.distance_m:.2f}" if result.distance_m is not None else "",
            result.crossed_speed,
            result.rear_speed,
        ]
        try:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except OSError:
            pass
        return True

    # ── Session management ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear deduplication set. Does NOT delete the file."""
        self._seen.clear()

    def new_session(self, path: Optional[str] = None) -> None:
        """
        Start a new session file with a timestamped name (or an explicit path).
        Clears the dedup set and ensures the header is written.
        """
        if path is None:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            path  = f"crossing_distances_{ts}.csv"
        self._path = path
        self._seen.clear()
        self._ensure_header()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_header(self) -> None:
        if not os.path.exists(self._path):
            try:
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.HEADERS)
            except OSError:
                pass
