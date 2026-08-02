"""Shared output paths for the figures pipeline.

All scripts in this folder write to figures/ (whole figures) and
figures/panels/ (individual panels for assembly).
"""
import os
from pathlib import Path

FIG_DIR   = Path(__import__("os").environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2])) / 'figures'
FIG_PANEL_DIR = FIG_DIR / 'panels'

FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PANEL_DIR.mkdir(parents=True, exist_ok=True)


def fig_out(name: str) -> str:
    return str(FIG_DIR / name)


def fig_panel(name: str) -> str:
    return str(FIG_PANEL_DIR / name)


def save_fig(fig, base_name: str, dpi_png: int = 300):
    """Save figure as pdf, png, svg in FIG_DIR with given base name."""
    base = FIG_DIR / base_name
    for ext in ('.pdf', '.png', '.svg'):
        kw = dict(bbox_inches='tight', pad_inches=0.06,
                  facecolor='white', transparent=False)
        if ext == '.png':
            kw['dpi'] = dpi_png
        fig.savefig(str(base) + ext, **kw)
    print(f"  ✓ {base}.pdf / .png / .svg")


def save_fig_panel(panel_fn, base_name: str, figsize):
    """Save a single panel function to the panels directory."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    panel_fn(ax)
    fig.tight_layout(pad=0.4)
    base = FIG_PANEL_DIR / base_name
    for ext in ('.pdf', '.png', '.svg'):
        kw = dict(bbox_inches='tight', pad_inches=0.06,
                  facecolor='white', transparent=False)
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(str(base) + ext, **kw)
    plt.close(fig)
    print(f"    panel: {base}.pdf")
