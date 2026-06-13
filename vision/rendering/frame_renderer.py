"""OpenCV drawing helper — vehicle boxes + ID badges.

In-ROI vehicles are drawn solid; vehicles outside the lane are drawn faint (or
hidden) so the operator's attention stays on the measured lane.
"""
import cv2
import numpy as np


def draw_tracks(frame: np.ndarray, tracks: list, inside_ids,
                speeds: dict | None = None,
                show_outside: bool = True) -> np.ndarray:
    inside = set(inside_ids or [])
    speeds = speeds or {}
    for t in tracks:
        x, y, w, h = (int(t.bbox[0]), int(t.bbox[1]),
                      int(t.bbox[2]), int(t.bbox[3]))
        is_in = t.id in inside
        if not is_in and not show_outside:
            continue
        clr = (0, 220, 80) if is_in else (110, 110, 110)
        cv2.rectangle(frame, (x, y), (x + w, y + h), clr,
                      2 if is_in else 1, cv2.LINE_AA)
        if is_in:
            spd = speeds.get(t.id, 0.0)
            label = f"#{t.id}  {spd:.0f} km/h" if spd > 1 else f"#{t.id}"
            (iw, ih), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ix = x + w // 2 - iw // 2
            iy = max(ih + 6, y - 4)
            cv2.rectangle(frame, (ix - 3, iy - ih - 3), (ix + iw + 3, iy + 3),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (ix, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)
    return frame
