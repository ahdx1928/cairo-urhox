#!/usr/bin/env python3
"""Shaded top-down render of the terrain, to sanity-check the shape."""
import struct

import numpy as np
from PIL import Image

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.path.join(ROOT, ".tmp", "cairo")
os.makedirs(TMP_DIR, exist_ok=True)
P = os.path.join(ROOT, "assets", "City", "cairo_terrain.bin")
OUT = os.path.join(TMP_DIR, "preview_terrain.png")

with open(P, "rb") as f:
    assert f.read(8) == b"URBXTRN1"
    ox, oz = struct.unpack("<ff", f.read(8))
    cell = struct.unpack("<f", f.read(4))[0]
    gw, gh = struct.unpack("<II", f.read(8))
    h = np.frombuffer(f.read(gw * gh * 2), dtype="<i2").reshape(gh, gw).astype(np.float64) * 0.1
    mat = np.frombuffer(f.read(gw * gh), dtype=np.uint8).reshape(gh, gw)

print("grid", gw, gh, "cell", cell)
print("height %.0f .. %.0f" % (h.min(), h.max()))

# hillshade from a north-west sun
dy, dx = np.gradient(h, cell)
nx = -dx
ny = np.full_like(h, 1.0)
nz = -dy
ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
nx, ny, nz = nx / ln, ny / ln, nz / ln
SUN = np.array([-0.5, 0.62, -0.6])
SUN = SUN / np.linalg.norm(SUN)
lam = np.clip(nx * SUN[0] + ny * SUN[1] + nz * SUN[2], 0, 1)

PAL = {
    0: np.array([196, 168, 122]),   # sand
    1: np.array([126, 112, 100]),   # rock
    2: np.array([30, 62, 96]),      # water
}
img = np.zeros((gh, gw, 3))
for k, c in PAL.items():
    m = mat == k
    img[m] = c
# shade land, leave water flat
land = mat != 2
img[land] *= (0.35 + 0.85 * lam[land])[:, None]
# snow caps
snow = (h > 1500) & land
img[snow] = np.array([236, 236, 240])

img = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(img).resize((900, 900), Image.LANCZOS).save(OUT)
print("wrote", OUT)
