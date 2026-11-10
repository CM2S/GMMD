"""Lay rendered panels out in rows, each cropped to its content and scaled to one height.

    python compose.py OUT.png --height 420 --gap 24 A.png B.png / C.png D.png

Panels are given row by row, the rows separated by "/". A 3D panel is cropped to the
RVE with a small margin; a 2D panel fills its image and is left as it is. The background
stays transparent, so the figure suits light and dark pages alike.
"""
import argparse

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("panels", nargs="+")
p.add_argument("--height", type=int, default=420)
p.add_argument("--gap", type=int, default=24)
a = p.parse_args()


def panel(path, height):
    im = Image.open(path).convert("RGBA")
    ys, xs = np.where(np.asarray(im)[:, :, 3] > 0)
    pad = round(0.012 * max(im.size))
    im = im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                  min(xs.max() + pad + 1, im.width), min(ys.max() + pad + 1, im.height)))
    # Scaled with the colours premultiplied by their opacity, so that the transparent
    # pixels around a particle do not bleed their colour into its edge
    size = (round(im.width * height / im.height), height)
    return im.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")


rows = [[]]
for item in a.panels:
    if item == "/":
        rows.append([])
    else:
        rows[-1].append(panel(item, a.height))
widths = [sum(im.width for im in r) + a.gap * (len(r) - 1) for r in rows]
out = Image.new("RGBA", (max(widths), len(rows) * a.height + a.gap * (len(rows) - 1)),
                (0, 0, 0, 0))
for i, (r, w) in enumerate(zip(rows, widths)):
    # A row narrower than the widest has the difference spread between its gaps
    gap = a.gap if len(r) == 1 else a.gap + (out.width - w) / (len(r) - 1)
    x = 0.0
    for im in r:
        out.paste(im, (round(x), i * (a.height + a.gap)))
        x += im.width + gap
out.save(a.out, optimize=True)
print(a.out, out.size)
