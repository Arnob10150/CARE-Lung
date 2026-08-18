"""Detect touching or overlapping elements in a rendered figure.

Renders a figure PDF at high resolution and reports the horizontal bands of the
image that contain ink, separated by clear white rows. Adjacent elements that
should be visually distinct (axis label vs. legend) must appear as separate
bands with a gap of at least MIN_GAP_PT points between them.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DPI = 600
MIN_GAP_PT = 3.0          # points of clear white required between elements


def ink_bands(png_path, x0f=0.0, x1f=1.0):
    from PIL import Image
    im = Image.open(png_path).convert("L")
    a = np.asarray(im)
    h, w = a.shape
    a = a[:, int(w * x0f):int(w * x1f)]
    rows_with_ink = (a < 200).any(axis=1)
    bands, start = [], None
    for i, v in enumerate(rows_with_ink):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(rows_with_ink) - 1))
    return bands, h


def main(pdf):
    tmp = Path(tempfile.gettempdir())
    png_stem = tmp / (Path(pdf).stem + "_gapchk")
    subprocess.run(["pdftoppm", "-png", "-r", str(DPI), pdf, str(png_stem)], check=True)
    png = sorted(tmp.glob(png_stem.name + "*.png"))[0]

    px_per_pt = DPI / 72.0
    print(f"{Path(pdf).name} at {DPI} dpi  ({px_per_pt:.2f} px/pt)")
    ok = True
    for label, (x0, x1) in [("left half", (0.0, 0.5)), ("right half", (0.5, 1.0))]:
        bands, h = ink_bands(png, x0, x1)
        print(f"\n  {label}: {len(bands)} ink band(s)")
        prev_end = None
        for (s, e) in bands:
            gap_pt = ((s - prev_end) / px_per_pt) if prev_end is not None else None
            gap_s = f"gap {gap_pt:5.1f} pt" if gap_pt is not None else "          "
            flag = ""
            if gap_pt is not None and gap_pt < MIN_GAP_PT:
                flag = "  <-- TOO TIGHT"
                ok = False
            print(f"    rows {s:5d}-{e:5d}  height {(e-s)/px_per_pt:5.1f} pt  {gap_s}{flag}")
            prev_end = e
    print()
    print("RESULT:", "clear separation everywhere" if ok
          else f"elements closer than {MIN_GAP_PT} pt found")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
