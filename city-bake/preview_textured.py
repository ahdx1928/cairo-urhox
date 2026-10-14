#!/usr/bin/env python3
"""Software rasteriser for the baked Cairo mesh, WITH texture sampling.

The runtime has no GPU in this sandbox, so this is the only way to actually see
what the scene looks like. It mirrors what the engine does:

  * reads the same cairo_mesh.bin (three vertex layouts)
  * samples the same texture PNGs through the baked UVs
  * applies a directional light matching the scene's sun

Not pixel-identical to the engine (no fog, no shadows, no HDR), but it answers
the question that matters: are the textures landing on the buildings?
"""
import math
import struct
import sys
import zlib

import numpy as np
from PIL import Image

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY_DIR = os.path.join(ROOT, "assets", "City")
TMP_DIR = os.path.join(ROOT, ".tmp", "cairo")
os.makedirs(TMP_DIR, exist_ok=True)

MESH = os.path.join(CITY_DIR, "cairo_mesh.bin")
TEX_DIR = CITY_DIR + os.sep
OUT = os.path.join(TMP_DIR, "preview_tex.png")

ANCHOR_LON, ANCHOR_LAT = 31.2357, 30.0455
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(ANCHOR_LAT))
WIN_MIN_LON, WIN_MAX_LON = 31.212, 31.268
WIN_MIN_LAT, WIN_MAX_LAT = 30.018, 30.066

TARGET = (((WIN_MIN_LON + WIN_MAX_LON) * 0.5 - ANCHOR_LON) * M_PER_DEG_LON,
          30.0,
          ((WIN_MIN_LAT + WIN_MAX_LAT) * 0.5 - ANCHOR_LAT) * M_PER_DEG_LAT)
YAW, PITCH, DIST, FOV = -0.55, -0.60, 420.0, 52.0

W, H = 1100, 620

FACADES = {
    0: "tex_wall_a.png",
    1: "tex_wall_b.png",
    2: "tex_wall_c.png",
    3: "tex_wall_d.png",
}
ROOF_STYLE = 100

# sun direction (towards the light); matches the scene's late-afternoon setup
SUN = np.array([0.42, 0.72, 0.55])
SUN = SUN / np.linalg.norm(SUN)

SKY_TOP = (30, 58, 110)
SKY_HORIZON = (150, 132, 108)


def load_mesh():
    """-> list of (verts, idx, layout, style). verts is (n, k) float32 plus uv."""
    chunks = []
    with open(MESH, "rb") as f:
        assert f.read(8) == b"URBXMSH1"
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            vb, ib, nv, ni, layout, style, _r = struct.unpack("<IIIIIII", f.read(28))
            struct.unpack("<ffffff", f.read(24))
            vraw = f.read(vb)
            iraw = f.read(ib)
            stride = (16, 20, 32)[layout]
            arr = np.frombuffer(vraw, dtype=np.uint8).reshape(nv, stride)
            pos = arr[:, 0:12].copy().view(np.float32).reshape(nv, 3)
            if layout == 0:
                col = arr[:, 12:15].astype(np.float32) / 255.0
                chunks.append((pos, None, None, col, layout, style,
                               np.frombuffer(iraw, dtype="<u2")))
            elif layout == 1:
                uv = arr[:, 12:20].copy().view(np.float32).reshape(nv, 2)
                chunks.append((pos, None, uv, None, layout, style,
                               np.frombuffer(iraw, dtype="<u2")))
            else:
                nrm = arr[:, 12:24].copy().view(np.float32).reshape(nv, 3)
                uv = arr[:, 24:32].copy().view(np.float32).reshape(nv, 2)
                chunks.append((pos, nrm, uv, None, layout, style,
                               np.frombuffer(iraw, dtype="<u2")))
    return chunks


def load_tex(name):
    im = Image.open(TEX_DIR + name).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    return a, a.shape[1], a.shape[0]


def sample(tex, w, h, u, v):
    """Nearest-neighbour sample with wrap. u,v are arrays of pixel coords."""
    ui = np.floor(u).astype(np.int64) % w
    vi = np.floor(v).astype(np.int64) % h
    return tex[vi, ui]


def main():
    chunks = load_mesh()
    print("chunks:", len(chunks))

    texcache = {}

    def get_tex(style, layout):
        key = (style, layout)
        if key not in texcache:
            if layout == 0:
                texcache[key] = None
            elif layout == 1:
                name = "tex_sand_island.png" if style == 1 else "tex_sand.png"
                texcache[key] = load_tex(name)
            else:
                name = "tex_roof.png" if style == ROOF_STYLE else FACADES.get(style, "tex_wall_a.png")
                texcache[key] = load_tex(name)
        return texcache[key]

    # camera basis
    cp, sp = math.cos(PITCH), math.sin(PITCH)
    cy, sy = math.cos(YAW), math.sin(YAW)
    eye = np.array([TARGET[0] - DIST * cp * sy,
                    TARGET[1] - DIST * sp,
                    TARGET[2] - DIST * cp * cy], dtype=np.float64)
    fwd = np.array(TARGET, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    camup = np.cross(right, fwd)

    focal = (H * 0.5) / math.tan(math.radians(FOV) * 0.5)
    cx, cyy = W * 0.5, H * 0.5

    zbuf = np.full((H, W), 1e30, dtype=np.float64)
    img = np.zeros((H, W, 3), dtype=np.float32)
    for y in range(H):
        t = y / float(H - 1)
        for k in range(3):
            img[y, :, k] = SKY_TOP[k] + (SKY_HORIZON[k] - SKY_TOP[k]) * t

    drawn = 0
    for pos, nrm, uv, col, layout, style, idx in chunks:
        tex = get_tex(style, layout)
        tri = idx.reshape(-1, 3)
        for t in tri:
            p = pos[t]
            # world -> camera
            rel = p - eye
            zc = rel @ fwd
            if np.any(zc <= 1.0):
                continue
            xc = rel @ right
            yc = rel @ camup
            sx = cx + xc * focal / zc
            syy = cyy - yc * focal / zc

            minx = max(0, int(sx.min()))
            maxx = min(W - 1, int(sx.max()) + 1)
            miny = max(0, int(syy.min()))
            maxy = min(H - 1, int(syy.max()) + 1)
            if minx > maxx or miny > maxy:
                continue

            x0, x1, x2 = sx
            y0, y1, y2 = syy
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-9:
                continue

            xs = np.arange(minx, maxx + 1) + 0.5
            ys = np.arange(miny, maxy + 1) + 0.5
            gx, gy = np.meshgrid(xs, ys)

            w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
            w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue

            iz = 1.0 / zc
            z = w0 * zc[0] + w1 * zc[1] + w2 * zc[2]

            # colour / texture
            if layout == 0:
                c = (w0[..., None] * col[t[0]] + w1[..., None] * col[t[1]]
                     + w2[..., None] * col[t[2]])
                rgb = c * 255.0
            else:
                # perspective-correct uv
                i0, i1, i2 = iz
                denom = w0 * i0 + w1 * i1 + w2 * i2
                uu = (w0 * i0 * uv[t[0], 0] + w1 * i1 * uv[t[1], 0]
                      + w2 * i2 * uv[t[2], 0]) / denom
                vv = (w0 * i0 * uv[t[0], 1] + w1 * i1 * uv[t[1], 1]
                      + w2 * i2 * uv[t[2], 1]) / denom
                tex_arr, tw, th = tex
                rgb = sample(tex_arr, tw, th, uu * tw, vv * th)
                if layout == 2:
                    n = (w0[..., None] * nrm[t[0]] + w1[..., None] * nrm[t[1]]
                         + w2[..., None] * nrm[t[2]])
                    nl = np.linalg.norm(n, axis=-1, keepdims=True)
                    n = n / np.where(nl < 1e-6, 1.0, nl)
                    lam = np.clip((n @ SUN)[..., None], 0.0, 1.0)
                    # ambient + sun, tuned so the texture is clearly readable
                    rgb = np.broadcast_to(rgb, n.shape) * (0.42 + 0.72 * lam)

            sub = zbuf[miny:maxy + 1, minx:maxx + 1]
            better = inside & (z < sub)
            if better.any():
                sub[better] = z[better]
                dst = img[miny:maxy + 1, minx:maxx + 1]
                if rgb.ndim == 2:
                    rgb = np.repeat(rgb[:, :, None], 3, axis=2)
                dst[better] = np.clip(rgb[better], 0, 255)
            drawn += 1

    print("triangles drawn:", drawn)
    out = np.clip(img, 0, 255).astype(np.uint8)

    raw = bytearray()
    for y in range(H):
        raw.append(0)
        raw.extend(out[y].tobytes())

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    blob += chunk(b"IEND", b"")
    with open(OUT, "wb") as fh:
        fh.write(blob)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
