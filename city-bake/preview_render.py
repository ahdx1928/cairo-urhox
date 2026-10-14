#!/usr/bin/env python3
"""Software rasterise the baked Cairo mesh from the game camera, as a
verification render. Not a pixel-accurate preview of the engine (no shadows,
fog or skybox), but it does confirm geometry, colours, winding and framing."""
import math
import struct
import sys
import zlib

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.path.join(ROOT, ".tmp", "cairo")
os.makedirs(TMP_DIR, exist_ok=True)
MESH = os.path.join(ROOT, "assets", "City", "cairo_mesh.bin")
OUT = os.path.join(TMP_DIR, "preview_render.png")

ANCHOR_LON, ANCHOR_LAT = 31.2357, 30.0455
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(ANCHOR_LAT))
WIN_MIN_LON, WIN_MAX_LON = 31.212, 31.268
WIN_MIN_LAT, WIN_MAX_LAT = 30.018, 30.066

TARGET = (((WIN_MIN_LON + WIN_MAX_LON) * 0.5 - ANCHOR_LON) * M_PER_DEG_LON,
          30.0,
          ((WIN_MIN_LAT + WIN_MAX_LAT) * 0.5 - ANCHOR_LAT) * M_PER_DEG_LAT)
YAW, PITCH, DIST, FOV = -0.55, -0.60, 1500.0, 52.0

W, H = 1280, 720
SKY_TOP = (26, 56, 118)
SKY_HORIZON = (199, 179, 148)


def load():
    chunks = []
    with open(MESH, "rb") as f:
        assert f.read(8) == b"URBXMSH1"
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            vbytes, ibytes, nv, ni = struct.unpack("<IIII", f.read(16))
            bb = struct.unpack("<ffffff", f.read(24))
            vraw = f.read(vbytes)
            iraw = f.read(ibytes)
            verts = []
            for i in range(nv):
                o = i * 16
                x, y, z = struct.unpack_from("<fff", vraw, o)
                r, g, b, a = struct.unpack_from("<BBBB", vraw, o + 12)
                verts.append((x, y, z, r, g, b))
            idx = struct.unpack("<%dH" % ni, iraw)
            chunks.append((verts, idx, bb))
    return chunks


def main():
    chunks = load()
    print("chunks:", len(chunks))

    cp, sp = math.cos(PITCH), math.sin(PITCH)
    cy, sy = math.cos(YAW), math.sin(YAW)
    eye = (TARGET[0] - DIST * cp * sy,
           TARGET[1] - DIST * sp,
           TARGET[2] - DIST * cp * cy)

    # camera basis: forward to target, right = fwd x up, up = right x fwd
    fx, fy, fz = (TARGET[0] - eye[0], TARGET[1] - eye[1], TARGET[2] - eye[2])
    fl = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / fl, fy / fl, fz / fl
    ux, uy, uz = 0.0, 1.0, 0.0
    rx = fy * uz - fz * uy
    ry = fz * ux - fx * uz
    rz = fx * uy - fy * ux
    rl = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rl, ry / rl, rz / rl
    vx = ry * fz - rz * fy
    vy = rz * fx - rx * fz
    vz = rx * fy - ry * fx

    f = (H * 0.5) / math.tan(math.radians(FOV) * 0.5)
    cx, cyy = W * 0.5, H * 0.5

    zbuf = [1e30] * (W * H)
    img = bytearray(W * H * 3)
    for y in range(H):
        t = y / float(H - 1)
        r = int(SKY_TOP[0] + (SKY_HORIZON[0] - SKY_TOP[0]) * t)
        g = int(SKY_TOP[1] + (SKY_HORIZON[1] - SKY_TOP[1]) * t)
        b = int(SKY_TOP[2] + (SKY_HORIZON[2] - SKY_TOP[2]) * t)
        base = y * W * 3
        for x in range(W):
            o = base + x * 3
            img[o], img[o + 1], img[o + 2] = r, g, b

    def project(p):
        dx, dy, dz = p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]
        zc = dx * fx + dy * fy + dz * fz
        if zc <= 0.5:
            return None
        xc = dx * rx + dy * ry + dz * rz
        yc = dx * vx + dy * vy + dz * vz
        return (cx + xc * f / zc, cyy - yc * f / zc, zc)

    drawn = 0
    for verts, idx, _ in chunks:
        for t in range(0, len(idx) - 2, 3):
            a = verts[idx[t]]
            b = verts[idx[t + 1]]
            c = verts[idx[t + 2]]
            pa = project(a)
            pb = project(b)
            pc = project(c)
            if not pa or not pb or not pc:
                continue
            xs = (pa[0], pb[0], pc[0])
            ys = (pa[1], pb[1], pc[1])
            minx = max(0, int(min(xs)))
            maxx = min(W - 1, int(max(xs)) + 1)
            miny = max(0, int(min(ys)))
            maxy = min(H - 1, int(max(ys)) + 1)
            if minx > maxx or miny > maxy:
                continue
            area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
            if abs(area) < 1e-9:
                continue
            for py in range(miny, maxy + 1):
                for px in range(minx, maxx + 1):
                    x = px + 0.5
                    y = py + 0.5
                    w0 = ((xs[1] - x) * (ys[2] - y) - (xs[2] - x) * (ys[1] - y)) / area
                    w1 = ((xs[2] - x) * (ys[0] - y) - (xs[0] - x) * (ys[2] - y)) / area
                    w2 = 1.0 - w0 - w1
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue
                    z = w0 * pa[2] + w1 * pb[2] + w2 * pc[2]
                    o = py * W + px
                    if z < zbuf[o]:
                        zbuf[o] = z
                        r = int(w0 * a[3] + w1 * b[3] + w2 * c[3])
                        g = int(w0 * a[4] + w1 * b[4] + w2 * c[4])
                        bb2 = int(w0 * a[5] + w1 * b[5] + w2 * c[5])
                        o3 = o * 3
                        img[o3] = min(255, r)
                        img[o3 + 1] = min(255, g)
                        img[o3 + 2] = min(255, bb2)
            drawn += 1

    print("triangles drawn:", drawn)

    raw = bytearray()
    for y in range(H):
        raw.append(0)
        raw.extend(img[y * W * 3:(y + 1) * W * 3])

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
