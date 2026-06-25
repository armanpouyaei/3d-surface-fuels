"""Portable Stage-1 model — train once on US sites, predict surface-fuel STRUCTURE
ANYWHERE from globally-free spaceborne inputs (AlphaEarth + Sentinel-1).

This powers the dashboard's "Generate anywhere" tab: no local LiDAR is needed at
inference, only the chosen AOI's AlphaEarth embeddings + Sentinel-1 (both global,
free). The model is trained on US 3DEP structure targets (`scripts/train_portable.py`)
and saved to disk; here we load it and run a single AOI on demand, with:
  * conformal prediction intervals (calibrated uncertainty),
  * an out-of-distribution (OOD) score per cell (distance to the training cloud) —
    so a location unlike the training ecosystems is flagged, not silently trusted,
  * a disk cache keyed by (lat, lon, size, year, res) for instant re-loads.

Honest scope: predictors and the trained relationship are global; the *target* (and
thus validation) is US-only. Off-distribution predictions are extrapolation — read
the OOD flag. Resolves to AlphaEarth's native 10 m; 1 m is mass-conserving
disaggregation with a fixed near-ground vertical profile.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np

from . import embeddings as emb, global30 as g30, sar
from .voxel import FuelVoxelGrid, GeoRef

_HERE = os.path.dirname(__file__)
PROC = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "processed"))
MODEL_PATH = os.path.join(PROC, "stage1_portable.joblib")
CACHE_DIR = os.path.join(PROC, "aoi_cache")

AEF_BANDS = [f"aef{i:02d}" for i in range(64)]
FEATS = AEF_BANDS + ["s1_vhvv", "s1_rvi"]
VPROFILE = np.array([0.45, 0.25, 0.13, 0.07, 0.05, 0.03, 0.015, 0.005], np.float32)  # near-ground shape

# ── memory guardrails (so a picked AOI can never OOM) ──────────────────────────
# Inference runs at 10 m, so cells scale as (size/10)². The cap below bounds every
# transient: at 2000 m → 200×200 = 40k 10-m cells (feature matrix ≈ 11 MB, AEF ≈ 10 MB).
# The only 1-m materialization is the optional NetCDF export, separately capped.
MAX_AOI_M = 2000.0            # hard cap on AOI side length (10-m grid stays ≤ 200×200)
MAX_1M_VOXELS = 30_000_000    # 1-m export budget (~120 MB float32); 1500 m @ nz=8 ≈ 18 M


def aoi_memory_estimate(size: float, res: float = 10.0, nz: int = 8) -> Dict[str, float]:
    """Rough footprint so the UI can show it before generating."""
    n10 = max(4, int(size // res))
    feat_mb = n10 * n10 * (len(FEATS) + 5) * 4 / 1e6     # feature matrix + AEF/S1 buffers
    vox_1m = int(size) * int(size) * nz
    return {"cells_10m": n10 * n10, "n10": n10, "predict_mb": round(feat_mb, 1),
            "voxels_1m": vox_1m, "export_1m_mb": round(vox_1m * 4 / 1e6, 1),
            "export_1m_ok": vox_1m <= MAX_1M_VOXELS}


# ── training (called by scripts/train_portable.py with pooled US samples) ──────
def train(X: np.ndarray, y: np.ndarray, sites: Optional[list] = None,
          out_path: str = MODEL_PATH, quantiles=g30.QUANTILES) -> str:
    """Fit quantile GBM + CQR + OOD stats on pooled (X, y); persist to ``out_path``."""
    import joblib
    rng = np.random.default_rng(0)
    cal = rng.random(len(y)) < 0.25
    models = g30._fit_quantiles(X[~cal], y[~cal], quantiles)
    delta = g30._cqr_delta(models, X[cal], y[cal], quantiles[0], quantiles[-1])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    # OOD on the 64 AlphaEarth bands ONLY — Sentinel-1 availability is flaky globally,
    # and zeroed S1 features would otherwise spuriously inflate the OOD distance.
    n_aef = 64
    Zaef = ((X - mu) / sd)[:, :n_aef]
    centroid = Zaef.mean(0)
    ood = np.sqrt(((Zaef - centroid) ** 2).sum(1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"models": models, "quantiles": quantiles, "delta": float(delta),
                 "mu": mu, "sd": sd, "centroid": centroid, "n_aef": n_aef,
                 "ood_thresh": float(np.percentile(ood, 95)), "feats": FEATS,
                 "sites": sites or [], "n_train": int(len(y))}, out_path)
    return out_path


def load() -> Optional[Dict]:
    if not os.path.exists(MODEL_PATH):
        return None
    import joblib
    return joblib.load(MODEL_PATH)


# ── geometry + features for an arbitrary AOI ──────────────────────────────────
def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


# ── FastFuels' surface layer = LANDFIRE FBFM40 → Scott & Burgan SB40 load lookup ──
# Total surface fuel load (tons/acre, summed over 1h+10h+100h+live-herb+live-woody),
# from RMRS-GTR-153 Table 7. FBFM40 raster codes: GR1-9=101-109, GS1-4=121-124,
# SH1-9=141-149, TU1-5=161-165, TL1-9=181-189; 91-99 = non-burnable (load 0).
_TON_ACRE_KG_M2 = 0.224170
_SB40_TONS = {
    "GR1": 0.40, "GR2": 1.10, "GR3": 2.00, "GR4": 2.15, "GR5": 2.90, "GR6": 3.50,
    "GR7": 6.40, "GR8": 8.80, "GR9": 11.00, "GS1": 1.35, "GS2": 2.60, "GS3": 3.25, "GS4": 12.80,
    "SH1": 1.95, "SH2": 8.35, "SH3": 9.65, "SH4": 4.75, "SH5": 8.60, "SH6": 5.75, "SH7": 14.40,
    "SH8": 10.65, "SH9": 15.50, "TU1": 3.70, "TU2": 4.20, "TU3": 3.25, "TU4": 6.50, "TU5": 14.00,
    "TL1": 6.80, "TL2": 5.90, "TL3": 5.50, "TL4": 6.20, "TL5": 8.05, "TL6": 4.80, "TL7": 9.80,
    "TL8": 8.30, "TL9": 14.10,
}
_PREFIX_BASE = {"GR": 100, "GS": 120, "SH": 140, "TU": 160, "TL": 180}
FBFM40_LOAD = {}  # numeric FBFM40 code -> total surface load (kg/m²)
for _name, _t in _SB40_TONS.items():
    FBFM40_LOAD[_PREFIX_BASE[_name[:2]] + int(_name[2:])] = round(_t * _TON_ACRE_KG_M2, 4)

# LANDFIRE FBFM40 ImageServer (free, no key) — available release years for vintage-matching.
_LF_YEARS = [2016, 2022, 2023, 2024]


def _lf_region(lon, lat):
    if -125 <= lon <= -66 and 24 <= lat <= 50:
        return "CONUS"
    if -179 <= lon <= -129 and 51 <= lat <= 72:
        return "AK"
    if -161 <= lon <= -154 and 18 <= lat <= 23:
        return "HI"
    if -68 <= lon <= -64 and 17 <= lat <= 19:
        return "PRVI"
    return None


def _lf_service(region, aef_year):
    """LANDFIRE FBFM40 service for the release year nearest the AlphaEarth year (tie → later)."""
    y = min(_LF_YEARS, key=lambda yy: (abs(yy - int(aef_year)), -yy))
    return f"Landfire_LF{y}/LF{y}_FBFM40_{region}", y


def _fetch_fbfm40(out, service):
    import io
    import requests
    import rasterio
    ny, nx = out["pred10"].shape
    x0, y0, res, epsg = float(out["x0"]), float(out["y0"]), float(out["res"]), int(out["epsg"])
    url = f"https://lfps.usgs.gov/arcgis/rest/services/{service}/ImageServer/exportImage"
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}, params={
        "bbox": f"{x0},{y0},{x0 + nx * res},{y0 + ny * res}", "bboxSR": epsg, "imageSR": epsg,
        "size": f"{nx},{ny}", "format": "tiff", "pixelType": "U16",
        "interpolation": "RSP_NearestNeighbor", "f": "image"})
    if r.status_code != 200 or "tiff" not in r.headers.get("content-type", ""):
        return None
    with rasterio.open(io.BytesIO(r.content)) as s:
        codes = s.read(1)                           # north-up (ArcGIS exportImage)
    load = np.zeros(codes.shape, np.float32)        # non-burnable / unmapped -> 0
    for code in np.unique(codes):
        if int(code) in FBFM40_LOAD:
            load[codes == code] = FBFM40_LOAD[int(code)]
    return load


def fastfuels_surface_aoi(out: Dict, lat: float, lon: float, year: int = 2024):
    """FastFuels' surface fuel LOAD (kg/m²) over the AOI: LANDFIRE FBFM40 → SB40 lookup,
    aligned to our grid (north-up), using the LANDFIRE release nearest ``year`` (the
    AlphaEarth vintage). Returns (load, lf_year) — (None, None) outside US / on failure.
    Uniform per 30 m fuel-model class — the baseline our heterogeneous product improves on."""
    region = _lf_region(lon, lat)
    if region is None:
        return None, None
    svc, lfy = _lf_service(region, year)
    tried = []
    for service, yy in [(svc, lfy), (f"Landfire_LF2024/LF2024_FBFM40_{region}", 2024)]:
        if service in tried:
            continue
        tried.append(service)
        try:
            load = _fetch_fbfm40(out, service)
        except Exception:
            load = None
        if load is not None:
            return load, yy
    return None, None


# ── GLOBAL categorical baseline: ESA WorldCover (10 m, global) → fuel-load crosswalk ──
# Same methodology as FastFuels (a categorical land/fuel map → representative per-class
# load), but worldwide. Loads use representative SB40 fuel models (kg/m²) so units match
# the US FastFuels panel. A constructed, coarse baseline — the global analog to LANDFIRE.
WORLDCOVER_LOAD = {
    10: FBFM40_LOAD[161],  # tree cover    -> TU1 (timber-understory)  0.829
    20: FBFM40_LOAD[122],  # shrubland     -> GS2 (grass-shrub)        0.583
    30: FBFM40_LOAD[102],  # grassland     -> GR2                      0.247
    40: FBFM40_LOAD[101],  # cropland      -> GR1 (sparse/managed)     0.090
    50: 0.0,               # built-up      -> non-burnable
    60: 0.045,             # bare / sparse
    70: 0.0,               # snow / ice
    80: 0.0,               # permanent water
    90: FBFM40_LOAD[102],  # herbaceous wetland -> grass               0.247
    95: FBFM40_LOAD[161],  # mangrove      -> forest understory        0.829
    100: 0.05,             # moss / lichen
}


def global_surface_aoi(out: Dict, lat: float, lon: float):
    """GLOBAL categorical surface-load baseline (kg/m²) over the AOI from ESA WorldCover
    (10 m, Planetary Computer) via the land-cover→fuel crosswalk. North-up, aligned to our
    grid. The worldwide analog to FastFuels' LANDFIRE layer. None on failure."""
    try:
        import planetary_computer as pc
        import rasterio
        from rasterio.warp import reproject, Resampling
        from affine import Affine
        from pystac_client import Client
        from pyproj import Transformer
        ny, nx = out["pred10"].shape
        x0, y0, res, epsg = float(out["x0"]), float(out["y0"]), float(out["res"]), int(out["epsg"])
        tr = Transformer.from_crs(epsg, 4326, always_xy=True)
        xs, ys = tr.transform([x0, x0 + nx * res, x0, x0 + nx * res],
                              [y0, y0, y0 + ny * res, y0 + ny * res])
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        items = list(cat.search(collections=["esa-worldcover"], bbox=bbox).items())
        if not items:
            return None
        items.sort(key=lambda it: it.properties.get("start_datetime", ""), reverse=True)  # latest map
        cls = np.zeros((ny, nx), np.uint8)
        dst_t = Affine(res, 0, x0, 0, -res, y0 + ny * res)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES"):
            with rasterio.open(items[0].assets["map"].href) as src:
                reproject(rasterio.band(src, 1), cls, dst_transform=dst_t, dst_crs=f"EPSG:{epsg}",
                          resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        load = np.zeros((ny, nx), np.float32)
        for c, v in WORLDCOVER_LOAD.items():
            load[cls == c] = v
        return load
    except Exception:
        return None


# ── canonical global grid of L-metre square tiles (UTM-snapped) ───────────────
# A click snaps to its containing tile so the AOI is canonical and deterministic —
# the same tile always yields the same cache key (reusable, shareable downloads).
def snap_to_grid(lat: float, lon: float, L: float) -> Dict:
    """Snap (lat,lon) to the L-metre UTM tile that contains it. Returns the tile's
    center lat/lon (the deterministic AOI center), UTM epsg, origin, and integer id."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    e, n = fwd.transform(lon, lat)
    i, j = int(np.floor(e / L)), int(np.floor(n / L))
    x0, y0 = i * L, j * L
    clon, clat = inv.transform(x0 + L / 2.0, y0 + L / 2.0)
    cr = [inv.transform(x0, y0), inv.transform(x0 + L, y0), inv.transform(x0 + L, y0 + L), inv.transform(x0, y0 + L)]
    las = [c[1] for c in cr]; los = [c[0] for c in cr]
    return {"lat": clat, "lon": clon, "epsg": epsg, "x0": x0, "y0": y0, "i": i, "j": j, "L": L,
            "bounds": [[min(las), min(los)], [max(las), max(los)]]}


def viewport_grid(bounds: Dict, L: float, max_cells: int = 400):
    """L-metre UTM grid cells overlapping a Leaflet ``bounds`` dict. Returns a list of
    {bounds:[[s,w],[n,e]], i, j} for folium rectangles, or None if too many (zoom out)."""
    if not bounds or "_southWest" not in bounds:
        return None
    from pyproj import Transformer
    sw, ne = bounds["_southWest"], bounds["_northEast"]
    clat, clon = (sw["lat"] + ne["lat"]) / 2.0, (sw["lng"] + ne["lng"]) / 2.0
    epsg = utm_epsg(clon, clat)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = [], []
    for la in (sw["lat"], ne["lat"]):
        for lo in (sw["lng"], ne["lng"]):
            x, y = fwd.transform(lo, la); xs.append(x); ys.append(y)
    i0, i1 = int(np.floor(min(xs) / L)), int(np.floor(max(xs) / L))
    j0, j1 = int(np.floor(min(ys) / L)), int(np.floor(max(ys) / L))
    if (i1 - i0 + 1) * (j1 - j0 + 1) > max_cells:
        return None
    cells = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            x0, y0 = i * L, j * L
            cr = [inv.transform(x0, y0), inv.transform(x0 + L, y0),
                  inv.transform(x0 + L, y0 + L), inv.transform(x0, y0 + L)]
            las = [c[1] for c in cr]; los = [c[0] for c in cr]
            cells.append({"i": i, "j": j, "bounds": [[min(las), min(los)], [max(las), max(los)]]})
    return cells


def _aoi_grid(lat, lon, size, res) -> Tuple[FuelVoxelGrid, int]:
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    x0, y0 = cx - size / 2.0, cy - size / 2.0
    n = max(4, int(size // res))
    grid = FuelVoxelGrid(np.zeros((1, n, n), np.float32), dz=res, dy=res, dx=res,
                         georef=GeoRef(f"EPSG:{epsg}", x0, y0, res))
    return grid, epsg


def _features(grid, year) -> Tuple[Optional[np.ndarray], Tuple[int, int], bool]:
    aef = emb.aef_for_grid(grid, year=year)
    if aef is None:
        return None, (grid.ny, grid.nx), False
    s1 = sar.sar_features_for_grid(grid)
    ny, nx = grid.ny, grid.nx
    cols = [np.nan_to_num(aef[i]).ravel() for i in range(64)]
    if s1 is not None:
        cols += [np.nan_to_num(s1["vh_vv"]).ravel(), np.nan_to_num(s1["rvi"]).ravel()]
    else:
        cols += [np.zeros(ny * nx, np.float32), np.zeros(ny * nx, np.float32)]
    return np.stack(cols, 1).astype(np.float32), (ny, nx), s1 is not None


def _cache_key(lat, lon, size, year, res):
    return hashlib.md5(f"{lat:.4f}_{lon:.4f}_{size:.0f}_{year}_{res:.0f}".encode()).hexdigest()[:12]


def predict_aoi(lat: float, lon: float, size: float = 1200.0, year: int = 2022,
                res: float = 10.0, use_cache: bool = True) -> Dict:
    """Predict surface-fuel structure over a global AOI. Returns maps + OOD + provenance.
    Cached to disk by (lat, lon, size, year, res)."""
    if size > MAX_AOI_M:
        raise ValueError(f"AOI {size:.0f} m exceeds the {MAX_AOI_M:.0f} m memory-safety cap "
                         "(pick a smaller AOI; tile larger areas).")
    cpath = os.path.join(CACHE_DIR, f"aoi_{_cache_key(lat, lon, size, year, res)}.npz")
    if use_cache and os.path.exists(cpath):
        d = np.load(cpath, allow_pickle=True)
        out = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
        out["cached"] = True
        return out
    m = load()
    if m is None:
        raise RuntimeError("Portable model not found — run `python scripts/train_portable.py` first.")
    grid, epsg = _aoi_grid(lat, lon, size, res)
    X, (ny, nx), has_s1 = _features(grid, year)
    if X is None:
        raise RuntimeError(f"AlphaEarth unavailable for {year} at ({lat:.3f},{lon:.3f}). "
                           "Try a year in 2017-2024.")
    qs = m["quantiles"]
    med = m["models"][qs[len(qs) // 2]].predict(X)
    lo = m["models"][qs[0]].predict(X) - m["delta"]
    hi = m["models"][qs[-1]].predict(X) + m["delta"]
    n_aef = m.get("n_aef", 64)
    Zaef = ((X - m["mu"]) / m["sd"])[:, :n_aef]      # OOD on AlphaEarth bands only (robust)
    ood = np.sqrt(((Zaef - m["centroid"]) ** 2).sum(1))
    pred10 = np.clip(med, 0, None).reshape(ny, nx)
    out = {
        "pred10": pred10,
        "lower": np.clip(lo, 0, None).reshape(ny, nx),
        "upper": np.clip(hi, 0, None).reshape(ny, nx),
        "ood": ood.reshape(ny, nx), "ood_thresh": float(m["ood_thresh"]),
        "epsg": epsg, "x0": grid.georef.x0, "y0": grid.georef.y0, "res": res,
        "lat": lat, "lon": lon, "size": size, "year": year, "has_s1": has_s1,
        "frac_ood": float((ood > m["ood_thresh"]).mean()),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cpath, **out)
    out["cached"] = False
    return out


def display_grid(pred10: np.ndarray, res10: float = 10.0, nz: int = 8) -> FuelVoxelGrid:
    """3D voxel grid for display: distribute the 10 m predicted load over a near-ground
    vertical profile. Voxels are res10 m horizontal × 1 m vertical (0..nz m)."""
    prof = VPROFILE[:nz] / VPROFILE[:nz].sum()
    bd = (pred10[None, :, :] * prof[:, None, None]).astype(np.float32)
    return FuelVoxelGrid(bd, dz=1.0, dy=res10, dx=res10,
                         attrs={"scenario": "predicted_global", "note": "spaceborne-predicted structure"})


def to_netcdf_1m(out: Dict, path: str, nz: int = 8) -> str:
    """Mass-conserving disaggregation of the 10 m prediction to a 1 m³ Option-C NetCDF."""
    pred10 = out["pred10"]
    if pred10.shape[0] * pred10.shape[1] * int(out["res"]) ** 2 * nz > MAX_1M_VOXELS:
        raise ValueError(f"1 m export would exceed the {MAX_1M_VOXELS:,}-voxel memory cap "
                         "for this AOI — use a smaller AOI.")
    up = np.repeat(np.repeat(pred10, int(out["res"]), 0), int(out["res"]), 1)  # 10 m -> 1 m load
    prof = VPROFILE[:nz] / VPROFILE[:nz].sum()
    bd = (up[None, :, :] * prof[:, None, None]).astype(np.float32)
    FuelVoxelGrid(bd, dz=1.0, dy=1.0, dx=1.0,
                  georef=GeoRef(f"EPSG:{int(out['epsg'])}", float(out["x0"]), float(out["y0"]), 1.0),
                  attrs={"scenario": "predicted_global_1m",
                         "source": "portable Stage-1 (AlphaEarth+Sentinel-1) -> 10 m -> 1 m disaggregation",
                         "note": "structure predicted from spaceborne; 1 m vertical/horizontal disaggregated"}
                  ).to_netcdf(path)
    return path
