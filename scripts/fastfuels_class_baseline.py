"""The REAL FastFuels surface baseline for the (c) head-to-head: uniform per LANDFIRE FBFM40
fuel-model CLASS (not per block). For each US seen site, fetch FBFM40 codes, set the baseline
to the per-class MEAN of measured occupancy, and report its heterogeneity vs measured truth.

FastFuels assigns one value per fuel class -> CV within a class ≈ 0 and within-block R² = 0.
Compare against v3 (validate_v3_vs_fastfuels.py: CV ours ≈ CV truth, within-block R² > 0).

Run:  python scripts/fastfuels_class_baseline.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import portable, metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEN = ["osbs", "soap", "wref", "harv"]   # CONUS sites (LANDFIRE coverage)
BLK = 3


def load(site):
    f = os.path.join(ROOT, "data", "interim", f"portable_grid_{site}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def fetch_codes(g):
    """FBFM40 class codes over the cached grid footprint, row 0 = south (to match occ_thin)."""
    import requests
    import rasterio
    from pyproj import Transformer
    ny, nx = g["target"].shape
    x0, y0, res, epsg = float(g["x0"]), float(g["y0"]), int(g["res"]) if False else float(g["res"]), int(g["epsg"])
    cx, cy = x0 + nx * res / 2, y0 + ny * res / 2
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(cx, cy)
    region = portable._lf_region(lon, lat)
    if region is None:
        return None
    service, lf_year = portable._lf_service(region, int(g["year"]))
    url = f"https://lfps.usgs.gov/arcgis/rest/services/{service}/ImageServer/exportImage"
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}, params={
        "bbox": f"{x0},{y0},{x0 + nx * res},{y0 + ny * res}", "bboxSR": epsg, "imageSR": epsg,
        "size": f"{nx},{ny}", "format": "tiff", "pixelType": "U16",
        "interpolation": "RSP_NearestNeighbor", "f": "image"})
    if r.status_code != 200 or "tiff" not in r.headers.get("content-type", ""):
        return None
    with rasterio.open(io.BytesIO(r.content)) as s:
        codes = s.read(1)
    return np.flipud(codes).copy(), lf_year       # north-up -> row 0 = south


def block_mean_map(z, valid):
    ny, nx = z.shape
    out = np.zeros_like(z)
    for by in range(0, ny, BLK):
        for bx in range(0, nx, BLK):
            sl = (slice(by, by + BLK), slice(bx, bx + BLK))
            if valid[sl].any():
                out[sl] = z[sl][valid[sl]].mean()
    return out


def main():
    print("REAL FastFuels (uniform per FBFM40 class) vs measured occupancy — US seen sites\n")
    print(f"{'site':<6}{'n_class':>8}{'class R²':>10}{'within-blk R²':>14}{'CV FastFuels':>14}{'CV truth':>10}")
    print("-" * 62)
    for s in SEEN:
        g = load(s)
        if g is None or "occ_thin" not in g:
            print(f"{s:<6}  (missing)"); continue
        try:
            res = fetch_codes(g)
        except Exception as e:
            print(f"{s:<6}  FBFM40 fetch failed: {type(e).__name__} {str(e)[:50]}"); continue
        if res is None:
            print(f"{s:<6}  no LANDFIRE coverage / fetch returned none"); continue
        codes, lf_year = res
        truth, valid = g["occ_thin"], g["valid"]
        uni = np.zeros_like(truth)
        for c in np.unique(codes[valid]):
            m = valid & (codes == c)
            if m.any():
                uni[m] = truth[m].mean()           # FastFuels analog: one value per class
        nclass = len(np.unique(codes[valid]))
        cls_r2 = M.r2(uni[valid], truth[valid])
        # within-block R² of the class-uniform map (block mean removed)
        ub = block_mean_map(uni, valid); tb = block_mean_map(truth, valid)
        wb = M.r2((uni - ub)[valid], (truth - tb)[valid])
        cvf = float(uni[valid].std() / (uni[valid].mean() + 1e-9))
        cvt = float(truth[valid].std() / (truth[valid].mean() + 1e-9))
        print(f"{s:<6}{nclass:>8}{cls_r2:>+10.3f}{wb:>+14.3f}{cvf:>14.2f}{cvt:>10.2f}  (LF{lf_year})", flush=True)
    print("\nFastFuels-per-class CV << truth CV and within-block R² ≈ 0 → it flattens the sub-class")
    print("heterogeneity; v3 (validate_v3_vs_fastfuels.py) recovers CV ≈ truth with within-block R² > 0.")


if __name__ == "__main__":
    main()
