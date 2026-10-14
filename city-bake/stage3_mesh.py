#!/usr/bin/env python3
"""Stage 3: bake GPU-ready meshes.

Python does all the extrusion and triangulation, so the Lua runtime only has to
hand byte blobs to VertexBuffer/IndexBuffer. Vertex layout is fixed:

    offset 0   position   3 x f32
    offset 12  colour     4 x u8   (rgba, normalised)

Sun shading is baked into the vertex colour (face normal dotted against a fixed
sun direction), which keeps the vertex small and lets the whole city render
unlit with one material. One draw call per chunk.

Output
  assets/City/cairo_mesh.bin
    char[8] "URBXMSH1"
    u32 nChunks
    per chunk:
      u32 vbytes, u32 ibytes
      u32 nVerts, u32 nIndices
      f32 minX,minY,minZ, maxX,maxY,maxZ
      u8[vbytes], u8[ibytes]
"""
import math
import os
import struct
import sys

# All paths hang off the repo root (the folder containing city-bake/), so the
# pipeline runs unchanged wherever the project is cloned.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".tmp", "cairo")
os.makedirs(TMP, exist_ok=True)

RAW = os.path.join(TMP, "cairo_raw.bin")
WATER_SHP = os.path.join(TMP, "gis_osm_water_a_free_1.shp")
ROADS_SHP = os.path.join(TMP, "gis_osm_roads_free_1.shp")
OUT = os.path.join(ROOT, "assets", "City", "cairo_mesh.bin")

LON_MIN, LON_MAX = 31.212, 31.268
LAT_MIN, LAT_MAX = 30.018, 30.066
LON0, LAT0 = 31.2357, 30.0455

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))

MIN_AREA = float(os.environ.get("MIN_AREA", "28"))
SMALL_AREA = float(os.environ.get("SMALL_AREA", "120"))
BIG_AREA = 500.0
DP_TOL = float(os.environ.get("DP_TOL", "2.0"))
MAX_RING = int(os.environ.get("MAX_RING", "24"))

FLOOR_H = 3.15
FLOOR_CAP = 30.0
CHUNK_VERT_LIMIT = 60000

# Layer heights. The whole ordering is:
#   plateau -1.20  <  ground -1.00  <  Nile -0.85  <  islands -0.70
#   <  roads -0.62  <  walls base -0.55
# Three constraints drive this:
#  * Everything must sit ABOVE the terrain plateau, or the desert sand buries
#    it (the plateau was once +2.0, which is what made roads and river vanish).
#  * The river must sit ABOVE the ground plane, or the continuous ground sheet
#    hides the Nile (WATER_Y used to be -1.20, below GROUND_Y).
#  * Roads must stay ABOVE the river so the bridge crossings read as bridges,
#    and the river's wave trough must not dip below GROUND_Y, or the river
#    blinks into sand (its material uses a small wave_height for that).
# Gaps of 0.10-0.20 m survive depth precision out to a few kilometres; the
# camera default distance is 420 m.
BASE_Y = -0.55      # building wall base
GROUND_Y = -1.00    # bare ground: brown dirt between blocks
WATER_Y = -0.85     # Nile surface, above the ground so the river shows
                    # (river wave_height is 0.04, so the band stays -0.93..-0.77)
ISLAND_Y = -0.70    # Zamalek / Roda land inside the water, above the wave crest
ROAD_Y = -0.62      # road surface, above the river so bridges cross over it

DENS_CORES = [
    (31.2385, 30.0455, 1.00),
    (31.2245, 30.0580, 0.85),
    (31.2130, 30.0400, 0.80),
    (31.2300, 30.0300, 0.70),
]

# direction towards the sun; shading is baked with this at build time
SUN_DIR = (0.45, 0.80, -0.40)
SAND = (128, 112, 90)
ISLAND_SAND = (140, 125, 100)
WATER_RGB = (26, 52, 74)


def hash01(x, y, seed=0.0):
    h = math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453
    return h - math.floor(h)


def m_area(ring):
    a = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a * 0.5


def to_local(lon, lat):
    return (lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT


def dp_simplify(pts, tol):
    if len(pts) <= 4:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    tol2 = tol * tol
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        best, besti = -1.0, -1
        for i in range(i0 + 1, i1):
            px, py = pts[i]
            if seg2 <= 1e-18:
                d2 = (px - x0) ** 2 + (py - y0) ** 2
            else:
                t = ((px - x0) * dx + (py - y0) * dy) / seg2
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                qx, qy = x0 + t * dx, y0 + t * dy
                d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 > best:
                best, besti = d2, i
        if best > tol2 and besti > 0:
            keep[besti] = True
            stack.append((i0, besti))
            stack.append((besti, i1))
    out = [pts[i] for i in range(len(pts)) if keep[i]]
    return out if len(out) >= 3 else pts


def min_area_rect(ring):
    best = None
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        ux, uy = ex / L, ey / L
        vx, vy = -uy, ux
        minu = minv = 1e30
        maxu = maxv = -1e30
        for (px, py) in ring:
            du = px * ux + py * uy
            dv = px * vx + py * vy
            minu = min(minu, du)
            maxu = max(maxu, du)
            minv = min(minv, dv)
            maxv = max(maxv, dv)
        area = (maxu - minu) * (maxv - minv)
        if best is None or area < best[0]:
            best = (area, ux, uy, vx, vy, minu, maxu, minv, maxv)
    if best is None:
        return None
    _, ux, uy, vx, vy, minu, maxu, minv, maxv = best
    return [
        (minu * ux + minv * vx, minu * uy + minv * vy),
        (maxu * ux + minv * vx, maxu * uy + minv * vy),
        (maxu * ux + maxv * vx, maxu * uy + maxv * vy),
        (minu * ux + maxv * vx, minu * uy + maxv * vy),
    ]


def density(clon, clat):
    acc = 0.0
    for (clon0, clat0, w) in DENS_CORES:
        d = math.hypot((clon - clon0) * M_PER_DEG_LON, (clat - clat0) * M_PER_DEG_LAT)
        acc += w / (1.0 + (d / 1150.0) ** 1.7)
    return 0.60 + 0.95 * min(1.0, acc)


def make_height(area, clon, clat):
    floors = (area ** 0.28) / 1.05
    floors *= density(clon, clat)
    floors *= 0.85 + 0.35 * hash01(clon * 1000.0, clat * 1000.0, 3.0)
    floors = max(1.0, min(floors, FLOOR_CAP))
    return floors * FLOOR_H + 0.7


def make_color(floors, clon, clat):
    j = (hash01(clon * 700.0, clat * 700.0, 11.0) - 0.5) * 26.0
    if floors <= 2.4:
        if hash01(clon * 300.0, clat * 300.0, 5.0) < 0.42:
            r, g, b = 168.0, 132.0, 112.0
        else:
            r, g, b = 206.0, 190.0, 164.0
    elif floors <= 5.5:
        r, g, b = 196.0, 182.0, 158.0
    elif floors <= 9.5:
        r, g, b = 180.0, 172.0, 160.0
    elif floors <= 16.0:
        r, g, b = 152.0, 152.0, 154.0
    else:
        r, g, b = 116.0, 132.0, 150.0
    return (int(max(0.0, min(255.0, r + j))),
            int(max(0.0, min(255.0, g + j))),
            int(max(0.0, min(255.0, b + j))))


def triangulate(xs, zs):
    """Ear clipping for a simple CCW ring. Returns index triples (0-based)."""
    n = len(xs)
    idx = list(range(n))
    tris = []

    def inside(px, pz, ax, az, bx, bz, cx, cz):
        d1 = (px - bx) * (az - bz) - (ax - bx) * (pz - bz)
        d2 = (px - cx) * (bz - cz) - (bx - cx) * (pz - cz)
        d3 = (px - ax) * (cz - az) - (cx - ax) * (pz - az)
        neg = d1 < 0 or d2 < 0 or d3 < 0
        pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (neg and pos)

    guard = 0
    while len(idx) > 3 and guard < 8192:
        guard += 1
        found = False
        m = len(idx)
        for k in range(m):
            ia = idx[(k - 1) % m]
            ib = idx[k]
            ic = idx[(k + 1) % m]
            ax, az = xs[ia], zs[ia]
            bx, bz = xs[ib], zs[ib]
            cx, cz = xs[ic], zs[ic]
            cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
            if cross <= 1e-9:
                continue
            clean = True
            for ip in idx:
                if ip in (ia, ib, ic):
                    continue
                if inside(xs[ip], zs[ip], ax, az, bx, bz, cx, cz):
                    clean = False
                    break
            if clean:
                tris.append((ia, ib, ic))
                idx.pop(k)
                found = True
                break
        if not found:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    elif len(idx) > 3:
        for i in range(1, len(idx) - 1):
            tris.append((idx[0], idx[i], idx[i + 1]))
    return tris


class Chunk:
    """Vertex accumulator.

    Two layouts are produced. Building chunks carry a baked vertex colour
    (position + rgba). Ground/water chunks instead carry UVs and are rendered
    with a textured unlit technique, so they store position + uv and their
    texture scale in metres per UV unit.
    """

    __slots__ = ("verts", "idx", "minX", "minY", "minZ", "maxX", "maxY", "maxZ",
                 "layout", "uv_scale", "style")

    def __init__(self, layout=0, uv_scale=4.0, style=0):
        self.verts = bytearray()
        self.idx = bytearray()
        self.layout = layout
        self.uv_scale = uv_scale
        self.style = style
        self.minX = self.minY = self.minZ = 1e30
        self.maxX = self.maxY = self.maxZ = -1e30

    @property
    def textured(self):
        return self.layout == 1

    @property
    def vstride(self):
        # 0: position+rgba  = 12+4  = 16
        # 1: position+uv    = 12+8  = 20
        # 2: position+normal+uv = 12+12+8 = 32
        return (16, 20, 32)[self.layout]

    @property
    def nv(self):
        return len(self.verts) // self.vstride

    @property
    def ni(self):
        return len(self.idx) // 2

    def _bounds(self, x, y, z):
        if x < self.minX:
            self.minX = x
        if y < self.minY:
            self.minY = y
        if z < self.minZ:
            self.minZ = z
        if x > self.maxX:
            self.maxX = x
        if y > self.maxY:
            self.maxY = y
        if z > self.maxZ:
            self.maxZ = z

    def vtx(self, x, y, z, r, g, b):
        self.verts += struct.pack("<fff", x, y, z)
        self.verts += struct.pack("<BBBB", r, g, b, 255)
        self._bounds(x, y, z)

    def vtxUV(self, x, y, z, u, v):
        """Textured vertex: position + uv (layout 1)."""
        self.verts += struct.pack("<fff", x, y, z)
        self.verts += struct.pack("<ff", u, v)
        self._bounds(x, y, z)

    def vtxPNU(self, x, y, z, nx, ny, nz, u, v):
        """Lit textured vertex: position + normal + uv (layout 2)."""
        self.verts += struct.pack("<fff", x, y, z)
        self.verts += struct.pack("<fff", nx, ny, nz)
        self.verts += struct.pack("<ff", u, v)
        self._bounds(x, y, z)

    def uv(self, x, z):
        """World position -> texture UV at this chunk's metre scale."""
        return (x / self.uv_scale, z / self.uv_scale)

    def tri(self, a, b, c):
        self.idx += struct.pack("<HHH", a, b, c)


def shade(rgb, nx, ny, nz):
    """Bake a directional term into a base colour (unlit rendering)."""
    d = nx * SUN_DIR[0] + ny * SUN_DIR[1] + nz * SUN_DIR[2]
    k = 0.46 + 0.54 * (d if d > 0.0 else 0.0)
    return (int(min(255, rgb[0] * k)),
            int(min(255, rgb[1] * k)),
            int(min(255, rgb[2] * k)))


# Which facade texture a building uses, chosen per building so neighbouring
# blocks differ. Order matches the material table in main.lua.
STYLE_PLASTER, STYLE_BRICK, STYLE_CONCRETE, STYLE_GLASS = 0, 1, 2, 3
STYLE_COUNT = 4
# metres of wall per horizontal texture tile
U_METRES = 7.0
# metres of roof per texture tile
ROOF_METRES = 6.0


def building_style(floors, clon, clat):
    """Pick a facade variant. Tall blocks skew to concrete/glass, low-rise to
    plaster and brick, with a little hash so the city is not banded by height."""
    r = hash01(clon * 421.0, clat * 421.0, 17.0)
    if floors >= 13.0:
        return STYLE_GLASS if r < 0.55 else STYLE_CONCRETE
    if floors >= 7.0:
        if r < 0.40:
            return STYLE_CONCRETE
        if r < 0.72:
            return STYLE_PLASTER
        return STYLE_BRICK
    if r < 0.46:
        return STYLE_PLASTER
    if r < 0.82:
        return STYLE_BRICK
    return STYLE_CONCRETE


ROOF_STYLE = 100   # marks a chunk as roof geometry so main.lua picks the roof material
# Non-building chunk tags. Heights alone are not a reliable way to tell these
# apart, so each gets an explicit style the runtime can switch on.
STYLE_WATER = 200
STYLE_ISLAND = 201
STYLE_GROUND = 202
STYLE_ROAD = 203


def emit_building(wall_ch, roof_ch, ring, h, floors, style):
    """Extruded footprint. Ring is CCW in XZ.

    Walls go into wall_ch (one window tile per storey), the roof into roof_ch
    (plain roof texture). Keeping them in separate chunks lets each be drawn
    with its own material without needing two geometries per model.

    UVs are absolute rather than per-building, so facades and roofs tile
    continuously along a block instead of restarting at every footprint.
    """
    n = len(ring)
    if n < 3:
        return

    wall_v = (h - BASE_Y) / FLOOR_H          # one texture tile per floor

    # cumulative perimeter distance -> u, so the facade wraps continuously
    cum = [0.0]
    for i in range(n):
        x0, z0 = ring[i]
        x1, z1 = ring[(i + 1) % n]
        cum.append(cum[-1] + math.hypot(x1 - x0, z1 - z0))

    for i in range(n):
        x0, z0 = ring[i]
        x1, z1 = ring[(i + 1) % n]
        ex, ez = x1 - x0, z1 - z0
        L = math.hypot(ex, ez)
        if L < 1e-6:
            ex, ez, L = 1.0, 0.0, 1.0
        nx, nz = ez / L, -ex / L

        u0 = cum[i] / U_METRES
        u1 = cum[i + 1] / U_METRES

        v0 = wall_ch.nv
        wall_ch.vtxPNU(x0, BASE_Y, z0, nx, 0.0, nz, u0, 0.0)
        wall_ch.vtxPNU(x0, h, z0, nx, 0.0, nz, u0, wall_v)
        wall_ch.vtxPNU(x1, h, z1, nx, 0.0, nz, u1, wall_v)
        wall_ch.vtxPNU(x1, BASE_Y, z1, nx, 0.0, nz, u1, 0.0)
        wall_ch.tri(v0, v0 + 1, v0 + 2)
        wall_ch.tri(v0, v0 + 2, v0 + 3)

    rb = roof_ch.nv
    for (x, z) in ring:
        roof_ch.vtxPNU(x, h, z, 0.0, 1.0, 0.0, x / ROOF_METRES, z / ROOF_METRES)
    xs = [p[0] for p in ring]
    zs = [p[1] for p in ring]
    for (a, b2, c) in triangulate(xs, zs):
        # reversed: ring order is CCW seen from above, which faces down
        roof_ch.tri(rb + a, rb + c, rb + b2)


def load_raw():
    out = []
    with open(RAW, "rb") as fh:
        assert fh.read(8) == b"URBXRAW1"
        (count,) = struct.unpack("<I", fh.read(4))
        for _ in range(count):
            (nv,) = struct.unpack("<H", fh.read(2))
            lons = struct.unpack("<%df" % nv, fh.read(4 * nv))
            lats = struct.unpack("<%df" % nv, fh.read(4 * nv))
            out.append((lons, lats))
    return out


def bake_buildings():
    """Emit buildings grouped by facade style, one chunk run per style, so each
    run can be drawn with that style's material."""
    buckets = [[] for _ in range(STYLE_COUNT)]
    kept = 0
    boxed = 0

    in_water = 0
    for lons, lats in load_raw():
        n = len(lons)
        clon = sum(lons) / n
        clat = sum(lats) / n
        if not (LON_MIN <= clon <= LON_MAX and LAT_MIN <= clat <= LAT_MAX):
            continue
        ring = [to_local(lons[i], lats[i]) for i in range(n)]
        # the footprints dataset carries piers, dredgers and riverbank detail
        # that lands inside the Nile once the river is drawn: anything whose
        # centroid falls in open water goes, so nothing stands in the channel.
        cx = sum(p[0] for p in ring) / len(ring)
        cz = sum(p[1] for p in ring) / len(ring)
        if in_channel(cx, cz):
            in_water += 1
            continue
        area = abs(m_area(ring))
        if area < MIN_AREA:
            continue

        if area < SMALL_AREA:
            rect = min_area_rect(ring)
            if rect is not None:
                ring = rect
                boxed += 1
        elif area < BIG_AREA:
            ring = dp_simplify(ring, DP_TOL)

        if len(ring) < 3:
            continue
        if len(ring) > MAX_RING:
            step = len(ring) / float(MAX_RING)
            ring = [ring[int(i * step)] for i in range(MAX_RING)]
        if m_area(ring) < 0:
            ring = ring[::-1]

        h = make_height(area, clon, clat)
        style = building_style(h / FLOOR_H, clon, clat)
        buckets[style].append((ring, h))
        kept += 1

    chunks = []
    for style in range(STYLE_COUNT):
        wall_ch = Chunk(layout=2, style=style)
        roof_ch = Chunk(layout=2, style=ROOF_STYLE)
        for ring, h in buckets[style]:
            # 4 wall verts + 1 roof vert per footprint point, plus slack
            if wall_ch.nv + 5 * len(ring) + 12 > CHUNK_VERT_LIMIT:
                chunks.append(wall_ch)
                wall_ch = Chunk(layout=2, style=style)
            if roof_ch.nv + len(ring) + 4 > CHUNK_VERT_LIMIT:
                chunks.append(roof_ch)
                roof_ch = Chunk(layout=2, style=ROOF_STYLE)
            emit_building(wall_ch, roof_ch, ring, h, h / FLOOR_H, style)
        if wall_ch.nv > 0:
            chunks.append(wall_ch)
        if roof_ch.nv > 0:
            chunks.append(roof_ch)

    counts = [len(b) for b in buckets]
    print("buildings: %d kept (%d boxed, %d in river removed) styles %s -> %d chunks, %d verts"
          % (kept, boxed, in_water, counts, len(chunks), sum(c.nv for c in chunks)))
    return chunks


def clip_rect(pts, xmin, xmax, zmin, zmax):
    """Sutherland-Hodgman clip of a polygon in local metres against an
    axis-aligned rectangle. Without this the Nile's polygons, which run the
    length of Egypt, get emitted whole and swamp the scene."""
    def clip_edge(poly, axis, bound, keep_greater):
        if not poly:
            return poly
        out = []
        n = len(poly)
        for i in range(n):
            cur = poly[i]
            prv = poly[i - 1]
            ci = cur[axis] >= bound if keep_greater else cur[axis] <= bound
            pi = prv[axis] >= bound if keep_greater else prv[axis] <= bound
            if ci:
                if not pi:
                    d = cur[axis] - prv[axis]
                    t = 0.0 if abs(d) < 1e-15 else (bound - prv[axis]) / d
                    out.append((prv[0] + (cur[0] - prv[0]) * t,
                                prv[1] + (cur[1] - prv[1]) * t))
                out.append(cur)
            elif pi:
                d = cur[axis] - prv[axis]
                t = 0.0 if abs(d) < 1e-15 else (bound - prv[axis]) / d
                out.append((prv[0] + (cur[0] - prv[0]) * t,
                            prv[1] + (cur[1] - prv[1]) * t))
        return out

    poly = list(pts)
    for axis, bound, keep_greater in ((0, xmin, True), (0, xmax, False),
                                      (1, zmin, True), (1, zmax, False)):
        poly = clip_edge(poly, axis, bound, keep_greater)
        if len(poly) < 3:
            return []
    return poly


def window_bounds():
    """The data window in local metres, matching the building clip."""
    xmin = (LON_MIN - LON0) * M_PER_DEG_LON
    xmax = (LON_MAX - LON0) * M_PER_DEG_LON
    zmin = (LAT_MIN - LAT0) * M_PER_DEG_LAT
    zmax = (LAT_MAX - LAT0) * M_PER_DEG_LAT
    return xmin, xmax, zmin, zmax


def water_rings():
    """Yield (is_hole, ring_in_local_metres) for every water ring clipped to the
    city window. Shapefile convention: outer rings are clockwise (negative area),
    holes counter-clockwise."""
    import shapefile
    xmin, xmax, zmin, zmax = window_bounds()
    r = shapefile.Reader(WATER_SHP)
    for i in range(len(r)):
        try:
            sr = r.shapeRecord(i)
        except Exception:
            continue
        shp = sr.shape
        if shp.shapeType != 5 or not shp.points:
            continue
        bb = shp.bbox
        if bb[2] < LON_MIN or bb[0] > LON_MAX or bb[3] < LAT_MIN or bb[1] > LAT_MAX:
            continue
        parts = list(shp.parts) + [len(shp.points)]
        for k in range(len(parts) - 1):
            ring = shp.points[parts[k]:parts[k + 1]]
            if len(ring) < 4:
                continue
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue
            pts = [to_local(p[0], p[1]) for p in ring]
            a = m_area(pts)
            if abs(a) < 1.0:
                continue
            clipped = clip_rect(pts, xmin, xmax, zmin, zmax)
            if len(clipped) < 3:
                continue
            if abs(m_area(clipped)) < 1.0:
                continue
            yield (a > 0), clipped


def cap_ring(pts, limit=320):
    """Simplify a clipped ring and subsample it to a vertex cap.

    The Nile's riverbank polygon arrives with >13000 points; ear clipping is
    O(n^2) per ring and a fan across so many points was what flooded the whole
    city with water (see bake_water). 320 points keeps the river silhouette
    smooth at normal view distances."""
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) <= limit:
        return pts
    step = len(pts) / float(limit)
    return [pts[int(i * step)] for i in range(limit)]


def point_in_poly(x, z, poly):
    """Ray-cast point inclusion, order-agnostic."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if (zi > z) != (zj > z):
            t = (z - zi) / (zj - zi)
            if x < xi + t * (xj - xi):
                inside = not inside
        j = i
    return inside


_water_polys_cache = None


def get_water_polys():
    """Clipped, simplified, CCW-normalised water rings.

    Returns (outers, islands): outer river/lake boundaries and the island
    (hole) rings inside them, both CCW in (x, z) with positive area."""
    global _water_polys_cache
    if _water_polys_cache is None:
        outers = []
        islands = []
        for is_hole, pts in water_rings():
            p = dp_simplify(list(pts), 2.0)
            if m_area(p) < 0:
                p = p[::-1]
            p = cap_ring(p)
            if len(p) < 3 or abs(m_area(p)) < 1.0:
                continue
            (islands if is_hole else outers).append(p)
        _water_polys_cache = (outers, islands)
    return _water_polys_cache


def in_channel(x, z):
    """True when the point falls in open water (a river ring, minus islands)."""
    outers, islands = get_water_polys()
    hit = False
    for ring in outers:
        if point_in_poly(x, z, ring):
            hit = True
            break
    if not hit:
        return False
    for ring in islands:
        if point_in_poly(x, z, ring):
            return False
    return True


def triangulate_up(ch, pts):
    """Emit a CCW (x, z) ring as up-facing triangles via ear clipping.

    The old code used a triangle fan. For a polygon the shape of the Nile
    (a long, meandering strip) a fan from one vertex sweeps across every
    concave bend, so its triangles carpeted the entire bounding box: the
    whole city turned into water. Ear clipping stays inside the ring."""
    base = ch.nv
    for (x, z) in pts:
        yield base, x, z
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    for (a, b, c) in triangulate(xs, zs):
        # triangulate emits CCW (x, z) winds, which face up (see bake_islands'
        # original fan: CCW rings already faced up)
        ch.tri(base + a, base + b, base + c)


def bake_water():
    # Water stays a flat vertex colour: the asset library has no suitable water
    # texture, and a solid blue reads correctly for the Nile at this scale.
    ch = Chunk(layout=0, style=STYLE_WATER)
    outers, _islands = get_water_polys()
    rings = 0
    for pts in outers:
        for (i, x, z) in triangulate_up(ch, pts):
            ch.vtx(x, WATER_Y, z, *WATER_RGB)
        rings += 1
    print("water: %d rings, %d verts" % (rings, ch.nv))
    return [ch] if ch.nv else []


def bake_islands():
    # islands are seen from the same distance, so they need a similarly large
    # tile or they wash out
    ch = Chunk(layout=1, uv_scale=120.0, style=STYLE_ISLAND)
    _outers, islands = get_water_polys()
    n = 0
    for pts in islands:
        for (i, x, z) in triangulate_up(ch, pts):
            u, v = ch.uv(x, z)
            ch.vtxUV(x, ISLAND_Y, z, u, v)
        n += 1
    print("islands: %d rings, %d verts" % (n, ch.nv))
    return [ch] if ch.nv else []


def bake_ground(extent=9000.0):
    """Bare ground under the city.

    Subdivided into a grid rather than one giant quad, because a single quad
    tiling a 60 m texture across 18 km repeats 300 times and reads as floor
    tiles. Each cell additionally gets its own UV offset and a slight brightness
    shift, which breaks the grid up so the eye finds no repeating unit.
    """
    ch = Chunk(layout=1, uv_scale=60.0, style=STYLE_GROUND)
    CELL = 420.0                       # metres per ground cell
    n = int(extent * 2 / CELL)
    step = (extent * 2) / n

    def uvjitter(i, j):
        """Per-cell offset plus a brightness tint."""
        r = hash01(i * 7.13, j * 3.71, 91.0)
        r2 = hash01(i * 2.31, j * 5.77, 113.0)
        return r, r2

    for j in range(n):
        for i in range(n):
            x0 = -extent + i * step
            z0 = -extent + j * step
            x1 = x0 + step
            z1 = z0 + step
            r, r2 = uvjitter(i, j)
            # shift the tile origin by up to a full repeat in each axis
            ox = r * 60.0
            oz = r2 * 60.0
            v0 = ch.nv
            for (x, z) in ((x0, z0), (x1, z0), (x1, z1), (x0, z0), (x1, z1), (x0, z1)):
                u = (x + ox) / 60.0
                v = (z + oz) / 60.0
                ch.vtxUV(x, GROUND_Y, z, u, v)
            ch.tri(v0, v0 + 1, v0 + 2)
            ch.tri(v0 + 3, v0 + 4, v0 + 5)
    print("ground: %dx%d cells, %d verts" % (n, n, ch.nv))
    return [ch]


# Road classes that get drawn, with the half-width in metres. Values follow
# typical Cairo street dimensions: a 2-lane residential street is about 7 m
# kerb to kerb, a 3-lane artery about 11 m.
ROAD_WIDTHS = {
    "motorway": 9.0, "motorway_link": 5.0,
    "trunk": 8.0, "trunk_link": 5.0,
    "primary": 7.0, "primary_link": 4.5,
    "secondary": 6.0, "secondary_link": 4.0,
    "tertiary": 5.0, "tertiary_link": 3.5,
    "unclassified": 4.0, "residential": 3.5,
    "living_street": 3.0, "pedestrian": 3.0,
    "service": 2.5, "track": 2.0,
    "footway": 1.4, "path": 1.2, "steps": 1.2, "cycleway": 1.4,
}
# Every road here is asphalt or paving; the ground beneath is brown dirt, so a
# single dark grey reads correctly for the whole network.
ROAD_RGB = (74, 74, 76)
PAVING_RGB = (126, 120, 110)   # footpaths read as lighter paving stone
PAVED_CLASSES = ("footway", "path", "steps", "pedestrian", "cycleway")


def road_width(fclass):
    return ROAD_WIDTHS.get(fclass)


def bake_roads():
    """Extrude every OSM road centreline into a flat quad strip."""
    import shapefile

    ch = Chunk(layout=0, style=STYLE_ROAD)   # flat colour: roads need no texture
    xmin, xmax, zmin, zmax = window_bounds()
    r = shapefile.Reader(ROADS_SHP)
    names = [f[0] for f in r.fields[1:]]

    drawn = 0
    skipped = 0
    for i in range(len(r)):
        try:
            sr = r.shapeRecord(i)
        except Exception:
            continue
        shp = sr.shape
        if not shp.points or len(shp.points) < 2:
            continue
        bb = shp.bbox
        if bb[2] < LON_MIN or bb[0] > LON_MAX or bb[3] < LAT_MIN or bb[1] > LAT_MAX:
            continue
        d = dict(zip(names, sr.record))
        fclass = d.get("fclass")
        hw = road_width(fclass)
        if hw is None:
            skipped += 1
            continue
        # skip anything flagged as a tunnel: it is under the ground we draw
        if str(d.get("tunnel", "F")) in ("T", "True", "true"):
            skipped += 1
            continue

        rgb = PAVING_RGB if fclass in PAVED_CLASSES else ROAD_RGB

        parts = list(shp.parts) + [len(shp.points)]
        for k in range(len(parts) - 1):
            line = shp.points[parts[k]:parts[k + 1]]
            if len(line) < 2:
                continue
            pts = [to_local(p[0], p[1]) for p in line]
            # clip: keep only segments that touch the window
            for j in range(len(pts) - 1):
                ax, az = pts[j]
                bx, bz = pts[j + 1]
                if (max(ax, bx) < xmin or min(ax, bx) > xmax
                        or max(az, bz) < zmin or min(az, bz) > zmax):
                    continue
                ex, ez = bx - ax, bz - az
                L = math.hypot(ex, ez)
                if L < 1e-4:
                    continue
                # perpendicular offset, so the segment becomes a quad
                px, pz = -ez / L * hw, ex / L * hw

                v0 = ch.nv
                ch.vtx(ax + px, ROAD_Y, az + pz, *rgb)
                ch.vtx(bx + px, ROAD_Y, bz + pz, *rgb)
                ch.vtx(bx - px, ROAD_Y, bz - pz, *rgb)
                ch.vtx(ax - px, ROAD_Y, az - pz, *rgb)
                # CCW seen from above so the normal points up
                ch.tri(v0, v0 + 1, v0 + 2)
                ch.tri(v0, v0 + 2, v0 + 3)
                drawn += 1

                if ch.nv + 8 > CHUNK_VERT_LIMIT:
                    yield ch
                    ch = Chunk(layout=0, style=STYLE_ROAD)

    if ch.nv > 0:
        yield ch
    print("roads: %d segments drawn (%d classes skipped)" % (drawn, skipped))


def write_file(chunks):
    with open(OUT, "wb") as fh:
        fh.write(b"URBXMSH1")
        fh.write(struct.pack("<I", len(chunks)))
        for c in chunks:
            # layout: 0 = position+rgba, 1 = position+uv, 2 = position+normal+uv
            # style indexes the facade material table (buildings only)
            fh.write(struct.pack("<IIIIIII", len(c.verts), len(c.idx), c.nv, c.ni,
                                 c.layout, c.style, 0))
            fh.write(struct.pack("<ffffff", c.minX, c.minY, c.minZ, c.maxX, c.maxY, c.maxZ))
            fh.write(bytes(c.verts))
            fh.write(bytes(c.idx))
    size = os.path.getsize(OUT)
    nv = sum(c.nv for c in chunks)
    ni = sum(c.ni for c in chunks)
    ntex = sum(1 for c in chunks if c.layout != 0)
    print("wrote %s: %d chunks (%d textured), %d verts, %d tris, %.2f MB"
          % (OUT, len(chunks), ntex, nv, ni // 3, size / 1048576.0))
    print("  bbox y: %.2f .. %.2f" % (min(c.minY for c in chunks), max(c.maxY for c in chunks)))


def main():
    chunks = []
    chunks += bake_buildings()
    chunks += bake_water()
    chunks += bake_islands()
    chunks += bake_ground()
    # roads are emitted last so their geometry is appended after the ground,
    # though ordering does not matter: they sit at their own height
    for rc in bake_roads():
        chunks.append(rc)
    write_file(chunks)


if __name__ == "__main__":
    main()
