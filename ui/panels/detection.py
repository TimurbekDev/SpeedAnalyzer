"""Detection settings panel — ROI, counting line, camera model, AI model."""

import tkinter as tk
from tkinter import ttk
from ui.theme.dark import Theme
from ui.widgets.components import GlowButton, section_label, divider, param_row, toggle_row


class DetectionPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=Theme.BG, **kwargs)
        self.app = app
        self._build()

    def _build(self):
        section_label(self, "DETECTION SETTINGS")

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=10)

        left  = tk.Frame(body, bg=Theme.BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = tk.Frame(body, bg=Theme.BG)
        right.pack(side="right", fill="both", expand=True, padx=(6, 0))

        self._card(left, "REGION OF INTEREST", [
            ("Set ROI",   Theme.CYAN,   self.app._roi_start),
            ("Clear All", Theme.DANGER, self.app._clear_all),
        ])

        line_card = tk.Frame(left, bg=Theme.SURFACE, padx=14, pady=12)
        line_card.pack(fill="x", pady=6)
        section_label(line_card, "COUNTING LINE", bg=Theme.SURFACE)
        divider(line_card, bg=Theme.SURFACE)
        tk.Frame(line_card, bg=Theme.SURFACE, height=4).pack()
        GlowButton(line_card, "Draw Line",  self.app._line_start,
                   accent=Theme.PURPLE, height=34).pack(fill="x", padx=14, pady=2)
        GlowButton(line_card, "Clear Line", self.app._clear_line,
                   accent=Theme.TEXT_3, height=34).pack(fill="x", padx=14, pady=2)

        hw_frame = tk.Frame(line_card, bg=Theme.SURFACE)
        hw_frame.pack(fill="x", padx=14, pady=(8, 2))
        tk.Label(hw_frame, text="LANE CORRIDOR WIDTH", bg=Theme.SURFACE,
                 fg=Theme.TEXT_3, font=Theme.F_CAP).pack(anchor="w")

        slider_row = tk.Frame(line_card, bg=Theme.SURFACE)
        slider_row.pack(fill="x", padx=14, pady=(2, 6))

        self._hw_lbl = tk.Label(slider_row,
                                text=f"±{self.app.var_lane_hw.get()} px",
                                bg=Theme.SURFACE, fg=Theme.CYAN,
                                font=Theme.F_MONO_S, width=8)
        self._hw_lbl.pack(side="right")

        hw_slider = ttk.Scale(
            slider_row, from_=20, to=300,
            orient="horizontal", variable=self.app.var_lane_hw,
            command=self._on_hw_change,
            style="Dark.Horizontal.TScale",
        )
        hw_slider.pack(side="left", fill="x", expand=True)

        tk.Label(line_card,
                 text="Tune so the blue corridor covers\nonly the target lane.",
                 bg=Theme.SURFACE, fg=Theme.TEXT_3,
                 font=Theme.F_CAP, justify="left").pack(anchor="w", padx=14, pady=(0, 4))

        cam_card = tk.Frame(right, bg=Theme.SURFACE, padx=14, pady=12)
        cam_card.pack(fill="x", pady=6)
        section_label(cam_card, "CAMERA MODEL", bg=Theme.SURFACE)
        divider(cam_card, bg=Theme.SURFACE)

        toggle_row(cam_card, "Enable Camera Model", self.app.var_cam_on,
                   command=self.app._apply_camera)
        param_row(cam_card, "Height (m)",  self.app.var_cam_H)
        param_row(cam_card, "Tilt (°)",    self.app.var_cam_T)
        param_row(cam_card, "Pan (°)",     self.app.var_cam_P)
        param_row(cam_card, "Focal (px)",  self.app.var_cam_F)

        tk.Frame(cam_card, bg=Theme.SURFACE, height=6).pack()
        GlowButton(cam_card, "Open Calibrator", self.app._open_calibrator,
                   accent=Theme.CYAN, height=34).pack(fill="x", padx=14, pady=2)
        GlowButton(cam_card, "Import JSON", self.app._load_cam_json,
                   accent=Theme.TEXT_3, height=34).pack(fill="x", padx=14, pady=2)

        ai_card = tk.Frame(right, bg=Theme.SURFACE, padx=14, pady=12)
        ai_card.pack(fill="x", pady=6)
        section_label(ai_card, "AI MODEL", bg=Theme.SURFACE)
        divider(ai_card, bg=Theme.SURFACE)

        model_row = tk.Frame(ai_card, bg=Theme.SURFACE)
        model_row.pack(fill="x", padx=0, pady=(8, 4))
        tk.Label(model_row, text="Active:", bg=Theme.SURFACE,
                 fg=Theme.TEXT_3, font=Theme.F_MONO_S).pack(side="left")
        self.lbl_model = tk.Label(model_row,
                                  text=self.app.model_name or "BG MODE",
                                  bg=Theme.SURFACE, fg=Theme.CYAN,
                                  font=Theme.F_MONO_S)
        self.lbl_model.pack(side="right")

        param_row(ai_card, "Confidence", self.app.var_conf)
        tk.Frame(ai_card, bg=Theme.SURFACE, height=4).pack()
        GlowButton(ai_card, "Load Custom Model", self.app._load_model,
                   accent=Theme.PURPLE, height=34).pack(fill="x", padx=14, pady=2)

    def _on_hw_change(self, _=None):
        hw = self.app.var_lane_hw.get()
        self._hw_lbl.config(text=f"±{hw} px")
        if self.app.cross_line.active:
            self.app.cross_line.set_width(hw)
            if self.app._last_frame is not None:
                self.app._render()

    def _card(self, parent, title, buttons):
        card = tk.Frame(parent, bg=Theme.SURFACE, padx=14, pady=12)
        card.pack(fill="x", pady=6)
        section_label(card, title, bg=Theme.SURFACE)
        divider(card, bg=Theme.SURFACE)
        tk.Frame(card, bg=Theme.SURFACE, height=6).pack()
        for text, accent, cmd in buttons:
            GlowButton(card, text, cmd, accent=accent,
                       height=34).pack(fill="x", padx=14, pady=2)
