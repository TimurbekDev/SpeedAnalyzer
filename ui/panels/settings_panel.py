"""Settings panel — visualization, BG subtractor, speed, stabilization."""

import tkinter as tk
from tkinter import ttk
from ui.theme.dark import Theme
from ui.widgets.components import section_label, divider, param_row, toggle_row


class SettingsPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.BG, **kwargs)
        self.app = app
        self._build()

    def _build(self):
        section_label(self, "SETTINGS")

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=10)

        left  = tk.Frame(body, bg=Theme.BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = tk.Frame(body, bg=Theme.BG)
        right.pack(side="right", fill="both", expand=True, padx=(6, 0))

        self._card(left, "VISUALIZATION", toggles=[
            ("Show Bounding Boxes", self.app.var_boxes),
            ("Show Speed Labels",   self.app.var_speed_lbl),
            ("Show Motion Trails",  self.app.var_trail),
        ])

        self._card(left, "BG SUBTRACTOR", params=[
            ("Min Area (px²)", self.app.var_min_a),
            ("Max Area (px²)", self.app.var_max_a),
        ])

        self._card(right, "YOLO MODEL", model_selector=[
            ("yolov8n", "Nano (fastest)"),
            ("yolov8s", "Small (balanced)"),
            ("yolov8m", "Medium (accurate)"),
            ("yolov8l", "Large (very accurate)"),
        ])

        self._card(right, "SPEED CONTROL", params=[
            ("Speed Limit (km/h)", self.app.var_limit),
            ("Speed Multiplier",   self.app.var_speed_mul),
            ("Calibration Factor", self.app.var_speed_k),
        ])

        self._card(right, "STABILIZATION", toggles=[
            ("Enable Stabilization", self.app.var_stab),
            ("Stabilize Viewport",   self.app.var_stabview),
        ])

    def _card(self, parent, title, toggles=None, params=None, model_selector=None):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="x", pady=6)
        section_label(card, title, bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)
        tk.Frame(card, bg=Theme.SURFACE, height=4).pack()
        for text, var in (toggles or []):
            toggle_row(card, text, var)
        for label, var in (params or []):
            param_row(card, label, var)
        if model_selector:
            self._model_selector(card, model_selector)
    
    def _model_selector(self, parent, models):
        """Dropdown to select YOLO model."""
        row = tk.Frame(parent, bg=Theme.SURFACE)
        row.pack(fill="x", pady=4)
        lbl = tk.Label(row, text="Model:", bg=Theme.SURFACE, fg=Theme.TEXT_1)
        lbl.pack(side="left", padx=(0, 8))
        
        models_dict = {name: model_id for model_id, name in models}
        current_model = self.app.model_name.replace(".pt", "") if self.app.model_name else "yolov8n"
        
        var = tk.StringVar(value=current_model)
        combo = tk.ttk.Combobox(row, textvariable=var, values=list(models_dict.keys()),
                                state="readonly", width=25)
        combo.pack(side="left", fill="x", expand=True)
        
        def on_select(event=None):
            selected = var.get()
            model_id = models_dict.get(selected, "yolov8n")
            self.app.reload_yolo(model_id)
        
        combo.bind("<<ComboboxSelected>>", on_select)
