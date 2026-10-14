#!/usr/bin/env python3
"""Generate the surface textures used by the Cairo scene.

Procedural on purpose: the official texture library resolves through a runtime
CDN whose file URLs are not fetchable here, so locally generated assets are the
only ones that can be verified end to end.

Facade tiles carry exactly one window each. The baker maps v to one tile per
floor and u to one tile per few metres of wall, so the window grid lines up with
real storeys and bays.

Outputs into assets/City/
  tex_wall_a..d.png   facade variants (plaster / brick / concrete / glass)
  tex_roof.png        plain roof surface (no windows)
  tex_sand.png        bare earth ground
  tex_sand_island.png riverbank earth
"""
import math
import os
import struct
import zlib

import numpy as np

# repo-relative so the pipeline runs wherever the project is cloned
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "City")
os.makedirs(OUT_DIR, exist_ok=True)
TILE = 256


# ---------------------------------------------------------------- noise ----

def value_noise(n, freq, seed):
    rng = np.random.default_rng(seed)
    g = rng.random((freq, freq))
    g = np.pad(g, ((0, 1), (0, 1)), mode="wrap")
    ys = np.linspace(0, freq, n, endpoint=False)
    xs = np.linspace(0, freq, n, endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[np.ix_(y0, x0)]
    b = g[np.ix_(y0, x0 + 1)]
    c = g[np.ix_(y0 + 1, x0)]
    d = g[np.ix_(y0 + 1, x0 + 1)]
    top = a * (1 - fx) + b * fx
    bot = c * (1 - fx) + d * fx
    return top * (1 - fy) + bot * fy


def fbm(n, seed, octaves=5):
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    freq = 4
    for i in range(octaves):
        out += amp * value_noise(n, freq, seed + i * 17)
        total += amp
        amp *= 0.5
        freq *= 2
    return out / total


# ------------------------------------------------------------- facades ----

def facade(base_rgb, glass_rgb, seed, style="plaster", tile=TILE):
    """One window per tile, centred, with wall margin on all sides."""
    n = tile
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    wall = fbm(n, seed, 5)
    fine = fbm(n, seed + 77, 4)

    rgb = np.zeros((n, n, 3), dtype=float)
    shade = 1.0 + 0.16 * (wall - 0.5) + 0.07 * (fine - 0.5)

    if style == "brick":
        # horizontal courses plus a running bond offset
        course = 16
        row = np.floor(yy / course).astype(int)
        off = (row % 2) * (course // 2)
        bx = np.floor((xx + off) / course).astype(int)
        by = row
        mortar = ((yy % course) < 1.6) | (((xx + off) % course) < 1.6)
        brick_tone = 1.0 + 0.13 * (((bx * 7 + by * 13) % 5) - 2) / 2.0
        shade = shade * brick_tone
        shade = np.where(mortar, shade * 0.80, shade)
    elif style == "concrete":
        # panel seams every 64 px
        seam = ((xx % 64) < 1.5) | ((yy % 64) < 1.5)
        shade = np.where(seam, shade * 0.86, shade)
    elif style == "glass":
        # curtain wall: strong vertical mullions
        mull = ((xx % 42) < 3.0)
        shade = np.where(mull, shade * 0.72, shade)

    for k in range(3):
        rgb[:, :, k] = base_rgb[k] * shade

    # ---- window ----
    wx0, wx1 = int(n * 0.24), int(n * 0.76)
    wy0, wy1 = int(n * 0.22), int(n * 0.74)

    if style == "glass":
        # glass towers: nearly the whole tile is glazing
        wx0, wx1 = int(n * 0.06), int(n * 0.94)
        wy0, wy1 = int(n * 0.08), int(n * 0.92)

    win_h = wy1 - wy0
    win_w = wx1 - wx0

    # vertical gradient across the pane: darker at the top
    gy = np.linspace(0.0, 1.0, win_h)[:, None]
    # a diagonal highlight band gives the pane some life
    gx = np.linspace(0.0, 1.0, win_w)[None, :]
    diag = np.clip(1.0 - np.abs((gx * 0.9 + gy * 0.35) - 0.42) * 2.6, 0.0, 1.0)

    pane = np.zeros((win_h, win_w, 3), dtype=float)
    for k in range(3):
        base = glass_rgb[k] * (0.62 + 0.55 * gy)
        pane[:, :, k] = base + 46.0 * diag
    # a little sky reflection variation
    pane += (fbm(win_h, seed + 5, 3)[:, :win_w, None] - 0.5) * 16.0 if win_h == win_w else 0.0

    rgb[wy0:wy1, wx0:wx1] = np.clip(pane, 0, 255)

    # frame around the pane
    fr = 3
    frame_col = np.array(base_rgb) * 1.16
    rgb[wy0 - fr:wy0, wx0 - fr:wx1 + fr] = frame_col
    rgb[wy1:wy1 + fr, wx0 - fr:wx1 + fr] = frame_col
    rgb[wy0 - fr:wy1 + fr, wx0 - fr:wx0] = frame_col
    rgb[wy0 - fr:wy1 + fr, wx1:wx1 + fr] = frame_col

    # sill: a slightly lighter, wider band under the window
    sill_col = np.array(base_rgb) * 1.22
    rgb[wy1 + fr:wy1 + fr + 4, max(0, wx0 - 8):min(n, wx1 + 8)] = sill_col

    return np.clip(rgb, 0, 255).astype(np.uint8)


def plain_surface(base_rgb, seed, grain=0.10, ripple=0.30, grit=0.006):
    n = TILE
    coarse = fbm(n, seed, 5)
    fine = fbm(n, seed + 101, 4)
    shade = 1.0 + ripple * (coarse - 0.5) + grain * (fine - 0.5)
    rgb = np.zeros((n, n, 3), dtype=float)
    for k in range(3):
        rgb[:, :, k] = base_rgb[k] * shade
    rng = np.random.default_rng(seed + 7)
    m = rng.random((n, n)) < grit
    rgb[m] *= 0.70
    return np.clip(rgb, 0, 255).astype(np.uint8)


def roof_surface(base_rgb, seed):
    """Gravel-covered flat roof with faint panel seams."""
    n = TILE
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    coarse = fbm(n, seed, 5)
    fine = fbm(n, seed + 101, 4)
    shade = 1.0 + 0.42 * (coarse - 0.5) + 0.20 * (fine - 0.5)
    # seams every 128 px
    seam = ((xx % 128) < 2.0) | ((yy % 128) < 2.0)
    shade = np.where(seam, shade * 0.80, shade)
    rgb = np.zeros((n, n, 3), dtype=float)
    for k in range(3):
        rgb[:, :, k] = base_rgb[k] * shade
    # gravel speckle, both light and dark
    rng = np.random.default_rng(seed + 7)
    r = rng.random((n, n))
    rgb[r < 0.020] *= 0.62
    rgb[r > 0.985] *= 1.20
    return np.clip(rgb, 0, 255).astype(np.uint8)


def rock_surface(base_rgb, seed):
    """Mountain rock: strata bands, cracks and coarse grain."""
    n = TILE
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    coarse = fbm(n, seed, 5)
    fine = fbm(n, seed + 101, 4)
    shade = 1.0 + 0.50 * (coarse - 0.5) + 0.22 * (fine - 0.5)
    # tilted strata, so the bands read as bedding planes on a slope
    band = np.sin((yy * 0.075 + xx * 0.030)) * 0.5 + 0.5
    shade *= 0.86 + 0.28 * band
    rgb = np.zeros((n, n, 3), dtype=float)
    for k in range(3):
        rgb[:, :, k] = base_rgb[k] * shade
    rng = np.random.default_rng(seed + 7)
    r = rng.random((n, n))
    rgb[r < 0.035] *= 0.58          # cracks
    rgb[r > 0.990] *= 1.25          # bright flecks
    return np.clip(rgb, 0, 255).astype(np.uint8)


def sand_surface(base_rgb, seed, ripple=0.35, grain=0.10, contrast=1.0):
    n = TILE
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # Four unrelated scales: broad patchiness, medium drift, fine grain, and a
    # faint diagonal. The previous version used two sine bands that were
    # obvious once tiled, which is what made the ground read as floor tiles.
    patch = fbm(n, seed, 3)          # very low frequency: big soft blotches
    drift = fbm(n, seed + 41, 5)     # medium: wind drift
    fine = fbm(n, seed + 101, 4)     # fine: grain
    micro = fbm(n, seed + 211, 3)    # micro: breaks up the flat areas
    # a couple of soft ridges, angled differently so they do not form a grid
    r1 = 0.5 + 0.5 * np.sin((xx * 0.021 + yy * 0.008) * 2.0 * math.pi / n)
    r2 = 0.5 + 0.5 * np.sin((xx * -0.006 + yy * 0.017) * 2.0 * math.pi / n)
    shade = (1.0
             + contrast * (0.55 * (patch - 0.5) + ripple * (drift - 0.5)
                           + grain * (fine - 0.5) + 0.06 * (micro - 0.5))
             + contrast * 0.10 * (r1 - 0.5) + contrast * 0.07 * (r2 - 0.5))
    rgb = np.zeros((n, n, 3), dtype=float)
    for k in range(3):
        rgb[:, :, k] = base_rgb[k] * shade
    rng = np.random.default_rng(seed + 7)
    m = rng.random((n, n)) < 0.006
    rgb[m] *= 0.72
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ png ----

def write_png(path, arr):
    h, w, _ = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(arr[y].tobytes())

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    blob += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(blob)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = []

    # facade variants: warm plaster, red brick, grey concrete, blue glass
    out.append(("tex_wall_a.png", facade((186, 166, 134), (58, 74, 92), 11, "plaster")))
    out.append(("tex_wall_b.png", facade((156, 108, 88), (54, 66, 82), 23, "brick")))
    out.append(("tex_wall_c.png", facade((158, 156, 150), (56, 68, 86), 37, "concrete")))
    out.append(("tex_wall_d.png", facade((104, 118, 134), (46, 62, 84), 51, "glass")))

    out.append(("tex_roof.png", roof_surface((150, 140, 124), 61)))
    out.append(("tex_sand.png", sand_surface((112, 98, 82), 11, ripple=0.46, grain=0.12, contrast=1.5)))
    # open desert: warmer, wind-rippled sand
    out.append(("tex_desert.png", sand_surface((186, 158, 112), 43, ripple=0.55, grain=0.11, contrast=1.7)))
    out.append(("tex_desert_b.png", sand_surface((172, 146, 106), 97, ripple=0.62, grain=0.13, contrast=1.8)))
    # mountain rock: banded strata with hard grain
    out.append(("tex_rock.png", rock_surface((128, 112, 98), 71)))
    out.append(("tex_sand_island.png", sand_surface((126, 112, 94), 29, ripple=0.30, grain=0.15, contrast=1.4)))

    for name, arr in out:
        p = os.path.join(OUT_DIR, name)
        write_png(p, arr)
        print("wrote %-26s %6.1f KB" % (name, os.path.getsize(p) / 1024.0))


if __name__ == "__main__":
    main()
