"""Vehicle — a single tracked vehicle for one frame.

A thin value object around a ByteTrack id + bbox.  All temporal logic (speed,
following gap) lives in the pipeline; this just exposes the geometry the rest of
the app needs (centre and ground-contact point).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vehicle:
    id:   int
    bbox: tuple   # (x, y, w, h) in video pixels

    @property
    def cx(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2.0

    @property
    def cy(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0

    @property
    def ground(self) -> tuple:
        """Bottom-centre — the wheels' contact with the road plane."""
        return (self.bbox[0] + self.bbox[2] / 2.0,
                self.bbox[1] + float(self.bbox[3]))
