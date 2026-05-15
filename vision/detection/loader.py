"""YOLO model loader — tries candidates in order, returns (model, name)."""
import os
from vision.detection.yolo_detector import YOLODetector


def load_yolo(path: str | None = None) -> tuple:
    """
    Try to load a YOLO model.  Returns (YOLODetector, model_name) or (None, None).
    Searches: explicit path → yolov8n → yolov8s → yolov9c → best → yolov10n.
    """
    try:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidates = ([path] if path else []) + [
            "yolov8n.pt", "yolov8s.pt", "yolov9c.pt", "best.pt", "yolov10n.pt",
        ]
        for c in candidates:
            if c and os.path.exists(c):
                try:
                    m = YOLO(c)
                    if device == "cuda":
                        m.to("cuda")
                    print(f"[YOLO] loaded {c} on {device}")
                    return YOLODetector(m), os.path.basename(c)
                except Exception as e:
                    print(f"[YOLO] {c}: {e}")
        # Last-resort: let ultralytics download yolov8n
        m = YOLO("yolov8n.pt")
        if device == "cuda":
            m.to("cuda")
        return YOLODetector(m), "yolov8n.pt"
    except Exception as e:
        print(f"[YOLO] not available: {e}")
        return None, None
