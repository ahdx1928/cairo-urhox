#!/usr/bin/env python3
"""Terrain ring around Cairo: desert flats, mountains, and the sea.

The city occupies a 5.4 x 5.3 km window. Everything outside it is generated
here as a heightfield, textured by slope and altitude:

    ocean      below sea level            water plane + sand seabed
    desert     near sea level, gentle     wind-rippled sand
    hills      mid slopes                 sand, darker in the hollows
    mountain   steep or high              banded rock
    snowline   the highest peaks          pale rock

Output: assets/City/cairo_terrain.bin
  char[8] "URBXTRN1"
  f32 originX, originZ, cellSize
  u32 gridW, gridH
  u16 heights[gridW*gridH]        metres, 0.1 m units
  u8  material[gridW*gridH]       0 sand, 1 rock, 2 water
"""
import math
import os
import struct

import numpy as np

# repo-relative so the pipeline runs wherever the project is cloned
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY_DIR = os.path.join(ROOT, "assets", "City")
os.makedirs(CITY_DIR, exist_ok=True)

OUT = os.path.join(CITY_DIR, "cairo_terrain.bin")
MESH_OUT = os.path.join(CITY_DIR, "cairo_terrain_mesh.bin")

# Chunk styles, matching the runtime material table. Terrain is split into one
# chunk group per material so each can use its own texture.
STYLE_TERRAIN_SAND = 300
STYLE_TERRAIN_ROCK = 301
STYLE_TERRAIN_WATER = 302

# Texture repeats, metres per tile. Must match scripts/city/CityTerrain.lua.
SAND_UV = 34.0
ROCK_UV = 26.0
TILE_CELLS = 64

# The city window, in the same local metres the mesh bake uses
CITY_HALF_X = 2900.0
CITY_HALF_Z = 2750.0
# gap kept clear between the city edge and the terrain, so nothing pops
CITY_MARGIN = 400.0

# terrain extent: a ring out to here
EXTENT = 22000.0
CELL = 110.0                    # metres per grid cell (mobile budget)

# The sea plane also carries the wave shader, whose crests rise ~1 m, so the
# waterline sits below the city plateau floor (-1.0): waves can never spill
# onto the built-up area even at a crest.
SEA_LEVEL = -2.4
SEABED_Y = -26.0                # where the shelf bottoms out

M_SAND, M_ROCK, M_WATER = 0, 1, 2


def fbm(shape, octaves, seed, base_freq, gain=0.5, lac=2.0):
    """Fractal value noise on a 2D grid, in [-1, 1]."""
    rng = np.random.default_rng(seed)
    h, w = shape
    out = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    total = 0.0
    freq = base_freq
    for _ in range(octaves):
        # lattice of random values, upsampled by cubic-ish interpolation
        gh = max(2, int(freq) + 1)
        gw = max(2, int(freq) + 1)
        g = rng.random((gh, gw))
        ys = np.linspace(0, gh - 1, h)
        xs = np.linspace(0, gw - 1, w)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, gh - 1)
        x1 = np.minimum(x0 + 1, gw - 1)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        # smoothstep
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        a = g[np.ix_(y0, x0)]
        b = g[np.ix_(y0, x1)]
        c = g[np.ix_(y1, x0)]
        d = g[np.ix_(y1, x1)]
        top = a * (1 - fx) + b * fx
        bot = c * (1 - fx) + d * fx
        out += amp * ((top * (1 - fy) + bot * fy) * 2.0 - 1.0)
        total += amp
        amp *= gain
        freq *= lac
    return out / total


def build():
    n = int(EXTENT * 2 / CELL) + 1
    print("grid: %d x %d  (%.0f m, %.0f m cells)" % (n, n, EXTENT * 2, CELL))

    xs = np.linspace(-EXTENT, EXTENT, n)
    zs = np.linspace(-EXTENT, EXTENT, n)
    gx, gz = np.meshgrid(xs, zs)

    # --- continental shape: sea to the north-west, mountains to the east ---
    # distance from the city, normalised
    d = np.sqrt(gx ** 2 + gz ** 2) / EXTENT
    # a directional bias so one side is sea and the other mountains
    sea_bias = (-gx * 0.55 - gz * 0.70) / EXTENT       # north-west gets positive
    mtn_bias = (gx * 0.75 - gz * 0.45) / EXTENT        # east-south-east

    # --- mountains: ridged noise gives sharp crests rather than blobs ---
    ridge = fbm((n, n), 6, 11, 3.0)
    ridge = 1.0 - np.abs(ridge)                        # ridge transform
    ridge = ridge ** 2.2
    mtn = ridge * (mtn_bias * 1.9 + 0.35)
    mtn = np.clip(mtn, 0.0, None)

    # --- dunes: low rolling noise for the desert floor ---
    dunes = fbm((n, n), 4, 29, 7.0)
    dune_h = dunes * 9.0 + fbm((n, n), 3, 37, 18.0) * 3.0

    # --- base elevation ---
    base = np.full((n, n), 26.0)
    base += dune_h * (1.0 - np.clip(mtn * 1.5, 0.0, 1.0))
    base += mtn * 1350.0

    # --- ocean: drop away where the sea bias is strong ---
    # The sea only starts well outside the city, so the desert ring around it
    # stays dry. Without this the shelf crept into the built-up area.
    shelf = np.clip(sea_bias * 1.5 - 0.10, 0.0, 1.0)
    shelf = shelf ** 1.7
    near_city = np.clip(1.0 - d * 2.1, 0.0, 1.0)
    near_city = near_city * near_city
    shelf *= (1.0 - near_city)
    base = base * (1.0 - shelf) + (SEABED_Y - 30.0 * shelf) * shelf

    # Flatten a plateau under the city so the baked buildings sit on level
    # ground, then ramp back up to the surrounding terrain. The ramp is long and
    # carries its own dune noise, otherwise the city reads as sitting on a
    # raised box.
    #
    # Height ordering (see stage3_mesh.py): the plateau must sit BELOW every
    # city layer (ground -0.80 ... islands -0.38) but ABOVE the waterline
    # (-2.4, plus ~1.1 m of wave crest), so the desert never buries the city
    # and the sea never floods it. It was +2.0 at one point, which buried the
    # roads, the ground sheet and the river under sand.
    PLATEAU_Y = -1.20
    RAMP = 3200.0
    fx = np.clip((np.abs(gx) - (CITY_HALF_X + CITY_MARGIN)) / RAMP, 0.0, 1.0)
    fz = np.clip((np.abs(gz) - (CITY_HALF_Z + CITY_MARGIN)) / RAMP, 0.0, 1.0)
    feather = np.maximum(fx, fz)
    # smoothstep so the join has no visible crease
    feather = feather * feather * (3.0 - 2.0 * feather)
    # ripples riding on the ramp, faded out at the flat end
    ramp_noise = fbm((n, n), 3, 53, 9.0) * 14.0 * feather * (1.0 - feather) * 4.0
    base = base * feather + PLATEAU_Y * (1.0 - feather) + ramp_noise

    # --- classify by height and slope ---
    dzdx = np.gradient(base, CELL, axis=1)
    dzdy = np.gradient(base, CELL, axis=0)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)             # rise over run

    mat = np.full((n, n), M_SAND, dtype=np.uint8)
    mat[slope > 0.85] = M_ROCK                          # steep faces are bare rock
    mat[base > 620.0] = M_ROCK                          # high ground loses its sand
    mat[base < SEA_LEVEL] = M_WATER

    # --- report ---
    land = base[mat != M_WATER]
    print("height: land %.0f .. %.0f m, mean %.0f m" % (
        land.min(), land.max(), land.mean()))
    print("slope:  p50 %.2f  p90 %.2f  max %.2f" % (
        np.percentile(slope, 50), np.percentile(slope, 90), slope.max()))
    for name, m in (("sand", M_SAND), ("rock", M_ROCK), ("water", M_WATER)):
        print("  %-6s %6.1f%%" % (name, 100.0 * (mat == m).mean()))

    # --- write ---
    hq = np.clip(base / 0.1, -32000, 32000).astype("<i2")
    with open(OUT, "wb") as f:
        f.write(b"URBXTRN1")
        f.write(struct.pack("<ff", -EXTENT, -EXTENT))
        f.write(struct.pack("<f", CELL))
        f.write(struct.pack("<II", n, n))
        f.write(hq.astype("<i2").tobytes())
        f.write(mat.tobytes())
    print("wrote %s (%.2f MB)" % (OUT, os.path.getsize(OUT) / 1048576.0))


# Sea surface plane, sitting at SEA_LEVEL. The terrain only supplies the
# seabed; without this the sea would render as bumpy dark ground rather than
# water, and it would not get the wave shader.
SEA_MESH_OUT = os.path.join(CITY_DIR, "cairo_sea_mesh.bin")
SEA_STYLE = 303


def bake_sea_mesh():
    """A flat grid covering the whole terrain extent at sea level.

    Only the part below the waterline shows, but covering the full extent keeps
    the edge from being visible when the camera swings around.
    """
    half = EXTENT
    n = 96                                  # cells per side
    step = (half * 2) / n
    verts = []
    for r in range(n + 1):
        z = -half + r * step
        for c in range(n + 1):
            x = -half + c * step
            verts.append((x, SEA_LEVEL, z, 0.0, 1.0, 0.0, x / 60.0, z / 60.0))
    idx = []
    stride = n + 1
    for r in range(n):
        for c in range(n):
            a = r * stride + c
            b = a + 1
            d = a + stride
            e = d + 1
            idx += [a, d, b, b, d, e]

    varr = np.array(verts, dtype="<f4")
    iarr = np.array(idx, dtype="<u2")
    with open(SEA_MESH_OUT, "wb") as f:
        f.write(b"URBXMSH1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<IIIIIII", varr.nbytes, iarr.nbytes,
                            len(verts), len(idx), 2, SEA_STYLE, 0))
        f.write(struct.pack("<ffffff", -half, SEA_LEVEL, -half,
                            half, SEA_LEVEL, half))
        f.write(varr.tobytes())
        f.write(iarr.tobytes())
    print("sea mesh: 1 chunk, %d verts, %.2f MB"
          % (len(verts), os.path.getsize(SEA_MESH_OUT) / 1048576.0))


def bake_mesh():
    """Turn the heightfield into GPU-ready chunks in the same format the city
    mesh loader already reads (URBXMSH1, layout 2 = position+normal+uv).

    Doing the triangulation here instead of at runtime keeps start-up fast: the
    runtime then only uploads byte blobs."""
    with open(OUT, "rb") as f:
        assert f.read(8) == b"URBXTRN1"
        ox, oz = struct.unpack("<ff", f.read(8))
        cell = struct.unpack("<f", f.read(4))[0]
        gw, gh = struct.unpack("<II", f.read(8))
        h = np.frombuffer(f.read(gw * gh * 2), dtype="<i2").reshape(gh, gw).astype(np.float64) * 0.1
        mat = np.frombuffer(f.read(gw * gh), dtype=np.uint8).reshape(gh, gw)

    # per-vertex normals from central differences
    dzdx = np.zeros_like(h)
    dzdy = np.zeros_like(h)
    dzdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2.0 * cell)
    dzdx[:, 0] = (h[:, 1] - h[:, 0]) / cell
    dzdx[:, -1] = (h[:, -1] - h[:, -2]) / cell
    dzdy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2.0 * cell)
    dzdy[0, :] = (h[1, :] - h[0, :]) / cell
    dzdy[-1, :] = (h[-1, :] - h[-2, :]) / cell

    nz_ = np.full_like(h, 1.0)
    nx_ = -dzdx
    ny_ = -dzdy
    ln = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2)
    nx_, ny_, nz_ = nx_ / ln, ny_ / ln, nz_ / ln

    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)
    rockT = np.clip((slope - 0.28) / 0.42, 0.0, 1.0)

    xs = ox + np.arange(gw) * cell
    zs = oz + np.arange(gh) * cell
    gx, gz = np.meshgrid(xs, zs)
    uvscale = 1.0 / (SAND_UV * (1.0 - rockT) + ROCK_UV * rockT)
    uu = gx * uvscale
    vv = gz * uvscale

    chunks = []
    n_tiles = 0
    for r0 in range(0, gh - 1, TILE_CELLS):
        for c0 in range(0, gw - 1, TILE_CELLS):
            r1 = min(r0 + TILE_CELLS, gh - 1)
            c1 = min(c0 + TILE_CELLS, gw - 1)
            cols = c1 - c0
            rows = r1 - r0
            if cols < 1 or rows < 1:
                continue
            nv = (cols + 1) * (rows + 1)
            verts = np.zeros((nv, 8), dtype=np.float32)
            k = 0
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    verts[k, 0] = gx[r, c]
                    verts[k, 1] = h[r, c]
                    verts[k, 2] = gz[r, c]
                    verts[k, 3] = nx_[r, c]
                    verts[k, 4] = ny_[r, c]
                    verts[k, 5] = nz_[r, c]
                    verts[k, 6] = uu[r, c]
                    verts[k, 7] = vv[r, c]
                    k += 1
            stride = cols + 1
            idx = []
            for r in range(rows):
                for c in range(cols):
                    a = r * stride + c
                    b = a + 1
                    d = a + stride
                    e = d + 1
                    idx += [a, d, b, b, d, e]
            idx = np.array(idx, dtype="<u2")

            # a tile is uniform material: pick the dominant class inside it
            block = mat[r0:r1 + 1, c0:c1 + 1]
            counts = [int((block == m).sum()) for m in (0, 1, 2)]
            dom = counts.index(max(counts))
            style = (STYLE_TERRAIN_SAND, STYLE_TERRAIN_ROCK, STYLE_TERRAIN_WATER)[dom]

            chunks.append({
                "style": style,
                "vbytes": verts.astype("<f4").tobytes(),
                "ibytes": idx.tobytes(),
                "nv": nv, "ni": len(idx),
                "min": (float(gx[r0, c0]), float(h[r0:r1 + 1, c0:c1 + 1].min()), float(gz[r0, c0])),
                "max": (float(gx[r1, c1]), float(h[r0:r1 + 1, c0:c1 + 1].max()), float(gz[r1, c1])),
            })
            n_tiles += 1

    with open(MESH_OUT, "wb") as f:
        f.write(b"URBXMSH1")
        f.write(struct.pack("<I", len(chunks)))
        for ch in chunks:
            f.write(struct.pack("<IIIIIII", len(ch["vbytes"]), len(ch["ibytes"]),
                                ch["nv"], ch["ni"], 2, ch["style"], 0))
            f.write(struct.pack("<ffffff", ch["min"][0], ch["min"][1], ch["min"][2],
                                ch["max"][0], ch["max"][1], ch["max"][2]))
            f.write(ch["vbytes"])
            f.write(ch["ibytes"])
    total_v = sum(c["nv"] for c in chunks)
    print("terrain mesh: %d chunks, %d verts, %.2f MB"
          % (n_tiles, total_v, os.path.getsize(MESH_OUT) / 1048576.0))


if __name__ == "__main__":
    build()
    bake_mesh()
    bake_sea_mesh()
