"""AlphaEarth Foundations embeddings — free, no Earth Engine.

Reads the public Source Cooperative mirror of the AlphaEarth annual Satellite
Embedding dataset (64-band, 10 m, UTM-zoned COGs, 2017-2025; CC-BY-4.0,
"produced by Google and Google DeepMind"). No account, no payment.

Access notes (from the bucket README):
* COGs are stored *bottom-up*; the sibling `.vrt` carries the correct top-down
  GeoTransform. We parse the `.vrt` XML to locate the tile (the `.tiff` filename
  is an opaque image hash, not a geocode), then reproject the raw `.tiff` window
  to the target grid (rasterio handles the bottom-up transform).
* De-quantize int8 -> float in [-1, 1]: ``((v/127.5)**2) * sign(v)``; NoData = -128.

`aef_for_grid(grid, year)` returns a (64, ny, nx) embedding stack aligned to a
`FuelVoxelGrid`'s footprint (mean-pooled 10 m -> grid resolution).
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np

S3 = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim")


def dequantize(v: np.ndarray) -> np.ndarray:
    """Raw int8 (-127..127, -128=NoData) -> float embedding in [-1, 1]."""
    v = v.astype(np.float32)
    out = ((v / 127.5) ** 2) * np.sign(v)
    out[v == -128] = np.nan
    return out


def _list_keys(prefix: str, suffix: str, session) -> list:
    keys, tok = [], None
    while True:
        u = f"{S3}?list-type=2&prefix={prefix}&max-keys=1000"
        if tok:
            import urllib.parse
            u += "&continuation-token=" + urllib.parse.quote(tok)
        x = ET.fromstring(session.get(u, timeout=30).content)
        keys += [c.find(f"{NS}Key").text for c in x.findall(f"{NS}Contents")
                 if c.find(f"{NS}Key").text.endswith(suffix)]
        t = x.find(f"{NS}NextContinuationToken")
        tok = t.text if t is not None else None
        if not tok:
            return keys


def _vrt_bounds(vrt_key: str, session):
    """Parse a tile's correct (top-down) bounds + EPSG from its .vrt XML."""
    txt = session.get(f"{S3}/{vrt_key}", timeout=30).text
    m = re.search(r"rasterXSize=\"(\d+)\" rasterYSize=\"(\d+)\"", txt)
    W, H = int(m.group(1)), int(m.group(2))
    gt = [float(x) for x in re.search(r"<GeoTransform>([^<]+)</GeoTransform>", txt).group(1).split(",")]
    epsg = int(re.search(r'AUTHORITY\["EPSG","(\d+)"\]\]\s*</SRS>', txt).group(1))
    left, xres, _, top, _, yres = gt
    right, bottom = left + W * xres, top + H * yres
    return (left, bottom, right, top), epsg


def find_tile(x, y, zone="16N", year=2024, cache=True) -> Optional[str]:
    """Find the AEF .tiff key whose footprint contains (x, y) in the zone CRS.
    Caches the (zone, year) tile->bounds index to disk so it's built once."""
    import requests
    s = requests.Session()
    os.makedirs(INDEX_DIR, exist_ok=True)
    idx_path = os.path.join(INDEX_DIR, f"aef_{year}_{zone}_index.json")
    index = json.load(open(idx_path)) if (cache and os.path.exists(idx_path)) else None

    if index is None:
        pfx = f"tge-labs/aef/v1/annual/{year}/{zone}/"
        vrts = _list_keys(pfx, ".vrt", s)
        index = {}
        for k in vrts:
            try:
                b, _ = _vrt_bounds(k, s)
                index[k.replace(".vrt", ".tiff")] = b
            except Exception:
                continue
        if cache:
            json.dump(index, open(idx_path, "w"))

    for tiff, b in index.items():
        if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
            return tiff
    return None


def aef_for_grid(grid, year=2024, zone=None) -> Optional[np.ndarray]:
    """Return a (64, ny, nx) dequantized AlphaEarth stack aligned to ``grid``
    (mean-pooled to the grid resolution). ``None`` if unavailable."""
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from affine import Affine
        from rasterio.windows import Window
        from rasterio.transform import rowcol

        g = grid.georef
        epsg = int(str(g.crs).split(":")[-1])
        zone = zone or _utm_zone_name(epsg)
        west, south = g.x0, g.y0
        east, north = g.x0 + grid.nx * grid.dx, g.y0 + grid.ny * grid.dy
        cx, cy = (west + east) / 2, (south + north) / 2

        tiff = find_tile(cx, cy, zone=zone, year=year)
        if tiff is None:
            return None
        dst = np.full((64, grid.ny, grid.nx), np.nan, np.float32)
        dst_t = Affine(grid.dx, 0, west, 0, -grid.dy, north)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES"):
            with rasterio.open("/vsicurl/" + S3 + "/" + tiff) as src:
                # orientation-agnostic window (the COG is bottom-up; rowcol handles it)
                rows, cols = rowcol(src.transform, [west, east, west, east],
                                    [south, south, north, north])
                r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
                win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
                arr = src.read(window=win, boundless=True, fill_value=-128)  # (64, h, w) int8
                src_t = src.window_transform(win)
                deq = dequantize(arr)
                deq = np.where(np.isfinite(deq), deq, 0.0)
                reproject(deq, dst, src_transform=src_t, src_crs=src.crs,
                          dst_transform=dst_t, dst_crs=f"EPSG:{epsg}",
                          resampling=Resampling.average)
        # dst is north-up (row 0 = north); flip to row 0 = south to match the rest
        # of the codebase (FuelVoxelGrid targets, sar.py, lidar.py all use row 0 = south).
        return dst[:, ::-1, :].copy()
    except Exception as e:  # noqa
        import sys
        print("aef_for_grid error:", repr(e)[:160], file=sys.stderr)
        return None


def _utm_zone_name(epsg: int) -> str:
    """EPSG:326NN -> 'NN N'; EPSG:327NN -> 'NN S' (AEF zone dir uses e.g. '16N')."""
    if 32601 <= epsg <= 32660:
        return f"{epsg - 32600}N"
    if 32701 <= epsg <= 32760:
        return f"{epsg - 32700}S"
    raise ValueError(f"not a UTM EPSG: {epsg}")
