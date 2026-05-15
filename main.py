#!/usr/bin/env python3
"""Entry point for Speed Analyzer."""

import tkinter as tk
from ui.app import SpeedAnalyzerModern


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = SpeedAnalyzerModern(root)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
