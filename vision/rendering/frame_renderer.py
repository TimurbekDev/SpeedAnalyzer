"""
OpenCV drawing helpers.

draw_tracks        — bounding boxes + ID badges for active tracks
draw_cross_flash   — fading speed+direction overlay at crossing moment
"""
import cv2
import numpy as np


def draw_tracks(frame: np.ndarray, tracks: list, scale: float,
                show_boxes: bool, show_speed: bool) -> np.ndarray:
    """Draw boxes and ID badges.  Speed text shown only via draw_cross_flash."""
    for t in tracks:
        x = int(t.bbox[0] * scale)
        y = int(t.bbox[1] * scale)
        w = int(t.bbox[2] * scale)
        h = int(t.bbox[3] * scale)
        clr = (0, 220, 80)

        if show_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), clr, 2, cv2.LINE_AA)

        if not show_speed:
            continue

        id_txt = f"#{t.id}"
        (iw, ih), _ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        ix = x + w // 2 - iw // 2
        iy = max(ih + 6, y - 4)
        cv2.rectangle(frame, (ix-3, iy-ih-3), (ix+iw+3, iy+3), (0, 0, 0), -1)
        cv2.putText(frame, id_txt, (ix, iy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1, cv2.LINE_AA)
    return frame


def draw_cross_flash(frame: np.ndarray, flash: dict, scale: float,
                     speed_limit: float, now: float, ttl: float = 2.5) -> None:
    """Draw fading speed+direction circles for recently-crossed vehicles."""
    stale = []
    for tid, info in flash.items():
        age = now - info["t0"]
        if age > ttl:
            stale.append(tid); continue

        cx    = int(info["cx"] * scale)
        cy    = int(info["cy"] * scale)
        spd   = info["speed"]
        direc = info["dir"]
        clr   = (0, 255, 200) if direc == "IN" else (0, 80, 255)
        alpha = max(0.0, 1.0 - age / ttl)
        radius = int(30 + 10 * alpha)

        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius, clr, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)

        spd_txt    = f"{direc}  {spd:.0f} km/h" if spd > 0 else direc
        over_limit = spd > 0 and spd > speed_limit
        txt_clr    = (0, 60, 255) if over_limit else clr
        (tw, th), _ = cv2.getTextSize(spd_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = cx - tw // 2
        ty = cy - radius - 8
        cv2.rectangle(frame, (tx-5, ty-th-4), (tx+tw+5, ty+4), (0, 0, 0), -1)
        cv2.putText(frame, spd_txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_clr, 2, cv2.LINE_AA)
        if over_limit:
            cv2.putText(frame, "! OVER LIMIT", (tx, ty + th + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 40, 255), 1, cv2.LINE_AA)

    for tid in stale:
        del flash[tid]
