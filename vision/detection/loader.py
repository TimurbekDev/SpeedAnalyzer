"""YOLO model loader — supports multiple model sizes with selection."""
import os
from vision.detection.yolo_detector import YOLODetector


AVAILABLE_MODELS = [
    ("yolov8n", "Nano (fastest)"),
    ("yolov8s", "Small (balanced)"),
    ("yolov8m", "Medium (accurate)"),
    ("yolov8l", "Large (very accurate)"),
]


def load_yolo(path: str | None = None, model_name: str | None = None) -> tuple:
    """
    Load a YOLO model.  Returns (YOLODetector, model_name_str) or (None, None).
    
    Args:
        path: explicit path to .pt file
        model_name: model id from AVAILABLE_MODELS (e.g. 'yolov8s')
                   If None, searches in order: path → local yolov8n.pt → download yolov8n
    """
    try:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # If specific model requested, try to load it (auto-download if needed)
        if model_name:
            try:
                print(f"[YOLO] Loading {model_name}...")
                m = YOLO(f"{model_name}.pt")
                if device == "cuda":
                    m.to("cuda")
                print(f"[YOLO] loaded {model_name} on {device}")
                return YOLODetector(m), f"{model_name}.pt"
            except Exception as e:
                print(f"[YOLO] {model_name}: {e}")
                return None, None
        
        # Fallback: try explicit path, then local files, then download
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
        
        # Last-resort: download yolov8n
        print("[YOLO] Downloading yolov8n...")
        m = YOLO("yolov8n.pt")
        if device == "cuda":
            m.to("cuda")
        return YOLODetector(m), "yolov8n.pt"
    except Exception as e:
        print(f"[YOLO] not available: {e}")
        return None, None
