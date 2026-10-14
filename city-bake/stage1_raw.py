#!/usr/bin/env python3
"""Stage 1: one slow pass over the Microsoft buildings tile.

Clips to a generous Cairo window and dumps raw footprint rings to an
intermediate file so later tuning passes are fast.

Intermediate (little-endian):
  char[8] "URBXRAW1"
  u32 count
  per building: u16 nverts, f32 lon[nverts], f32 lat[nverts]
"""
import gzip
import json
import os
import struct
import sys

# repo-relative so the pipeline runs wherever the project is cloned
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".tmp", "cairo")
os.makedirs(TMP, exist_ok=True)

SRC = os.path.join(TMP, "egypt_122121122.csv.gz")
DST = os.path.join(TMP, "cairo_raw.bin")

LON_MIN, LON_MAX = 31.185, 31.285
LAT_MIN, LAT_MAX = 29.995, 30.100


def main():
    n_scan = 0
    kept = 0
    with open(DST, "wb") as fh:
        fh.write(b"URBXRAW1")
        fh.write(struct.pack("<I", 0))
        with gzip.open(SRC, "rt", encoding="utf-8", errors="replace") as src:
            for line in src:
                n_scan += 1
                if n_scan % 2000000 == 0:
                    print("  scanned %d kept %d" % (n_scan, kept), file=sys.stderr, flush=True)
                if '"coordinates"' not in line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                g = o.get("geometry")
                if not g or g.get("type") != "Polygon":
                    continue
                rings = g["coordinates"]
                if not rings:
                    continue
                raw = rings[0]
                if len(raw) < 4:
                    continue
                ring = raw[:-1] if raw[0] == raw[-1] else raw
                if len(ring) < 3 or len(ring) > 65535:
                    continue
                s = float(len(ring))
                clon = sum(p[0] for p in ring) / s
                clat = sum(p[1] for p in ring) / s
                if not (LON_MIN <= clon <= LON_MAX and LAT_MIN <= clat <= LAT_MAX):
                    continue
                nv = len(ring)
                fh.write(struct.pack("<H", nv))
                fh.write(struct.pack("<%df" % nv, *[p[0] for p in ring]))
                fh.write(struct.pack("<%df" % nv, *[p[1] for p in ring]))
                kept += 1
        fh.seek(8)
        fh.write(struct.pack("<I", kept))
    print("raw scan done: scanned %d, kept %d, %.1f MB"
          % (n_scan, kept, os.path.getsize(DST) / 1048576.0))


if __name__ == "__main__":
    main()
