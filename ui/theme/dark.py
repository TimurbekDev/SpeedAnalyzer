"""Design system — colors, fonts, spacing."""


class Theme:
    BG       = "#080c14"
    SURFACE  = "#0d1526"
    SURFACE2 = "#111e30"
    SURFACE3 = "#172440"
    SIDEBAR  = "#090d1a"
    TOPBAR   = "#090d1a"

    CYAN         = "#00e5ff"
    CYAN_DIM     = "#006080"
    CYAN_GLOW    = "#00e5ff22"
    PURPLE       = "#7c3aed"
    PURPLE_DIM   = "#3d1a7a"

    SUCCESS  = "#00ff88"
    WARNING  = "#ffb800"
    DANGER   = "#ff3366"
    INFO     = "#0099ff"

    TEXT_1   = "#e2e8f0"
    TEXT_2   = "#94a3b8"
    TEXT_3   = "#475569"

    BORDER   = "#1e3a5f"
    DIVIDER  = "#0f1d30"

    SPEED_OK   = "#00ff88"
    SPEED_WARN = "#ffb800"
    SPEED_BAD  = "#ff3366"

    F_TITLE  = ("Segoe UI", 13, "bold")
    F_BODY   = ("Segoe UI", 10)
    F_SMALL  = ("Segoe UI", 9)
    F_MONO   = ("Consolas", 10)
    F_MONO_S = ("Consolas", 9)
    F_MONO_T = ("Consolas", 8)
    F_BIG    = ("Segoe UI", 28, "bold")
    F_MED    = ("Segoe UI", 18, "bold")
    F_CAP    = ("Consolas", 8, "bold")

    RADIUS   = 6
    PAD      = 12
    PAD_S    = 6

    @classmethod
    def speed_color(cls, speed: float, limit: float = 60.0) -> str:
        if speed < limit:
            return cls.SPEED_OK
        if speed < limit * 1.3:
            return cls.SPEED_WARN
        return cls.SPEED_BAD
