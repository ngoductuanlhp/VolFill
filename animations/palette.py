"""Manim palette for the project page. Mirrors css/tokens.css (keep in sync)."""
import os

THEMES = {
    "dark": dict(
        bg="#0E1015", surface="#161A22", surface2="#1C212B", border="#262B36", border_strong="#353B48",
        text="#E8EAED", muted="#9AA3B2", faint="#6B7383",
        blue="#58C4DD", teal="#5CD0B3", yellow="#F7D96F", red="#FC6255", green="#83C167", purple="#9A72AC", occluded="#6E8FC4",
    ),
    "light": dict(
        bg="#FAF9F6", surface="#FFFFFF", surface2="#F1EFEA", border="#E2DFD7", border_strong="#CFCBC1",
        text="#16181D", muted="#565D69", faint="#7D8491",
        blue="#1C758A", teal="#237A66", yellow="#B8860B", red="#C0392B", green="#3F7A26", purple="#6D4C7D", occluded="#4F6FA6",
    ),
}

THEME = os.environ.get("VOLFILL_THEME", "dark")
P = THEMES[THEME]


def rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
