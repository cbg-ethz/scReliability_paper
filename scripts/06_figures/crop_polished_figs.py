#!/usr/bin/env python3
"""Split a multi-page PowerPoint-exported PDF and crop each page top+bottom
only, preserving the original horizontal page width.

Why: pdfcrop --bbox auto-detects content extent on all four sides. If we
crop the left/right margins, each figure ends up with a slightly different
width, and \\includegraphics[width=\\textwidth]{...} scales each by a
different factor — fonts then appear at different sizes across figures.
By keeping the original page width and only trimming top/bottom whitespace,
the horizontal scale is identical across all figures.

Output: <SCRELIABILITY_FIG_OUT>/figN.pdf (default: cropped_figures/), one per page.
"""
from pathlib import Path
import numpy as np
import fitz

import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
SRC = Path(_os.environ.get("SCRELIABILITY_PPT_PDF", str(_pathlib.Path(_ROOT) / "scReliability_Figures.pdf")))
OUT_DIR = Path(_os.environ.get("SCRELIABILITY_FIG_OUT", str(SRC.parent / "cropped_figures")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARGIN_PT = 6           # padding to keep above/below content
WHITE_THRESH = 250      # RGB value above which pixel is considered "white"

doc = fitz.open(str(SRC))

for i in range(doc.page_count):
    page = doc[i]
    full_rect = page.rect

    # Render page at 72 dpi so 1 pixel == 1 PDF point.
    pix = page.get_pixmap(dpi=72, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    rgb = arr[:, :, :3] if pix.n >= 3 else arr[:, :, :1]

    # Non-white if any channel is below the threshold (catches anti-alias edges).
    is_content = (rgb < WHITE_THRESH).any(axis=2)
    rows_with_content = np.where(is_content.any(axis=1))[0]
    if len(rows_with_content) == 0:
        print(f'  page {i+1}: empty (skipped)')
        continue

    y_top = rows_with_content[0]
    y_bot = rows_with_content[-1]

    new_y0 = max(0, y_top - MARGIN_PT)
    new_y1 = min(full_rect.height, y_bot + 1 + MARGIN_PT)
    new_h = new_y1 - new_y0

    # Crop rectangle in PDF coordinates: keep original full width.
    crop_rect = fitz.Rect(0, new_y0, full_rect.width, new_y1)

    out = fitz.open()
    new_page = out.new_page(width=float(full_rect.width), height=float(new_h))
    new_page.show_pdf_page(new_page.rect, doc, i, clip=crop_rect)

    out_path = OUT_DIR / f'fig{i+1}.pdf'
    out.save(str(out_path))
    out.close()

    print(f'  fig{i+1}: {full_rect.width:.0f} x {new_h:.0f} pt '
          f'({full_rect.width / 72:.2f} x {new_h / 72:.2f} in)')

print(f'✓ Wrote {doc.page_count} cropped figures to {OUT_DIR}')
