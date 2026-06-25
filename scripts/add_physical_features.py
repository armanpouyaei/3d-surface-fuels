"""Add PHYSICAL sub-canopy features to each cached grid, to attack the within-biome residual:
  L-band SAR  (ALOS PALSAR annual mosaic, MPC, ungated) — HH/HV γ0 dB + HH−HV ratio.
              L-band penetrates canopy, so it carries woody-understory/stem info that
              optical (AlphaEarth) and C-band (Sentinel-1) cannot see. (The project's
              original radar premise.)
  terrain     (Copernicus DEM GLO-30, MPC) — elevation, slope, TPI. Topographic position
              drives understory moisture/light hence density.

Appends pal_hh, pal_hv, pal_ratio, elev, slope, tpi (each (ny,nx), row 0 = south to match
AEF/S1) to data/interim/portable_grid_*.npz. No LAZ re-read; ~10 s/site like the AEF/S1 fetch.

Run:  python scripts/add_physical_features.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MPC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def bbox_lonlat(x0, y0, nx, ny, res, epsg):
    from pyproj import Transformer
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = tr.transform([x0, x0 + nx * res, x0, x0 + nx * res],
                          [y0, y0, y0 + ny * res, y0 + ny * res])
    return min(xs), min(ys), max(xs), max(ys)


def fetch(d):
    import planetary_computer as pc
    import rasterio
    from rasterio.warp import reproject, Resampling
    from affine import Affine
    from pystac_client import Client

    x0, y0 = float(d["x0"]), float(d["y0"]); epsg = int(d["epsg"]); res = float(d["res"])
    ny, nx = d["target"].shape; year = int(d["year"])
    west, south, east, north = bbox_lonlat(x0, y0, nx, ny, res, epsg)
    cat = Client.open(MPC, modifier=pc.sign_inplace)
    dst_t = Affine(res, 0, x0, 0, -res, y0 + ny * res)
    dst_crs = f"EPSG:{epsg}"

    def read(href, resamp=Resampling.bilinear):
        out = np.full((ny, nx), np.nan, np.float32)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES"):
            with rasterio.open(href) as src:
                reproject(rasterio.band(src, 1), out, dst_transform=dst_t, dst_crs=dst_crs,
                          resampling=resamp, src_nodata=0, dst_nodata=np.nan)
        return out

    def finish(a):
        a = np.flipud(a).astype(np.float32)
        m = np.nanmedian(a)
        return np.where(np.isfinite(a), a, m if np.isfinite(m) else 0.0).astype(np.float32)

    # ---- L-band PALSAR: nearest available year ----
    pitems = list(cat.search(collections=["alos-palsar-mosaic"], bbox=[west, south, east, north]).items())
    if not pitems:
        raise RuntimeError("no PALSAR items")
    yr = min({i.datetime.year for i in pitems}, key=lambda y: abs(y - year))
    sel = [i for i in pitems if i.datetime.year == yr]
    hh_s, hv_s = [], []
    for it in sel:
        try:
            hh_s.append(read(it.assets["HH"].href)); hv_s.append(read(it.assets["HV"].href))
        except Exception:
            continue
    with np.errstate(invalid="ignore", divide="ignore"):
        hh_dn = np.nanmedian(np.stack(hh_s), 0); hv_dn = np.nanmedian(np.stack(hv_s), 0)
        hh_db = 20 * np.log10(np.where(hh_dn > 0, hh_dn, np.nan)) - 83.0   # JAXA γ0 cal
        hv_db = 20 * np.log10(np.where(hv_dn > 0, hv_dn, np.nan)) - 83.0
        ratio = hh_db - hv_db

    # ---- Copernicus DEM GLO-30: elevation -> slope, TPI ----
    ditems = list(cat.search(collections=["cop-dem-glo-30"], bbox=[west, south, east, north]).items())
    elev_s = [read(i.assets["data"].href) for i in ditems[:4]]
    elev = finish(np.nanmedian(np.stack(elev_s), 0)) if elev_s else np.zeros((ny, nx), np.float32)
    gy, gx = np.gradient(elev, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    tpi = (elev - uniform_filter(elev, 5)).astype(np.float32)

    return dict(pal_hh=finish(hh_db), pal_hv=finish(hv_db), pal_ratio=finish(ratio),
                elev=elev, slope=slope, tpi=tpi, pal_year=yr)


def main():
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        s = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = dict(np.load(f))
        if "pal_hv" in d:
            print(f"  [{s}] physical feats present — skip", flush=True); continue
        try:
            feats = fetch(d)
            d.update(feats)
            np.savez_compressed(f, **d)
            v = d["valid"]
            print(f"  [{s}] PALSAR({feats['pal_year']}) HV μ{feats['pal_hv'][v].mean():.1f}dB "
                  f"ratio μ{feats['pal_ratio'][v].mean():.1f} | slope μ{feats['slope'][v].mean():.1f}° "
                  f"tpi σ{feats['tpi'][v].std():.2f}", flush=True)
        except Exception as e:
            print(f"  [{s}] FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
