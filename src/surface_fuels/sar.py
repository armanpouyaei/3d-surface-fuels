"""Real Sentinel-1 SAR features for a fuel grid (Week 2).

Fetches analysis-ready **Sentinel-1 RTC** (radiometric-terrain-corrected gamma0
VV/VH backscatter) from Microsoft Planetary Computer — free, no credentials —
and derives the fusion-relevant features over a `FuelVoxelGrid`'s footprint:

  * VV, VH backscatter (gamma0, linear power)
  * VH/VV cross-pol ratio  — the understory-density / volume-scattering proxy
  * RVI = 4*VH / (VV + VH)  — radar vegetation index

A temporal median over several acquisitions suppresses speckle. SAR is ~10 m, so
it is resampled (bilinear) to the grid and is honestly a coarse covariate, not a
1 m sensor. Arrays are returned row 0 = south to match `FuelVoxelGrid` loads.
Returns ``None`` (never raises) if offline so callers can degrade gracefully.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _grid_bbox_lonlat(grid):
    from pyproj import Transformer
    g = grid.georef
    epsg = int(str(g.crs).split(":")[-1])
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = tr.transform(
        [g.x0, g.x0 + grid.nx * grid.dx, g.x0, g.x0 + grid.nx * grid.dx],
        [g.y0, g.y0, g.y0 + grid.ny * grid.dy, g.y0 + grid.ny * grid.dy])
    return min(xs), min(ys), max(xs), max(ys), epsg


def sar_features_for_grid(grid, start="2025-09-01", end="2026-06-30",
                          max_items=10) -> Optional[dict]:
    """Median-composite Sentinel-1 RTC over the grid footprint -> feature dict
    (vv, vh, vh_vv, rvi), each (ny, nx) aligned to the grid (row 0 = south)."""
    try:
        import planetary_computer as pc
        import rasterio
        from rasterio.warp import reproject, Resampling
        from affine import Affine
        from pystac_client import Client

        west, south, east, north, epsg = _grid_bbox_lonlat(grid)
        cat = Client.open(MPC_STAC, modifier=pc.sign_inplace)
        items = list(cat.search(
            collections=["sentinel-1-rtc"], bbox=[west, south, east, north],
            datetime=f"{start}/{end}", query={"sar:instrument_mode": {"eq": "IW"}},
        ).items())[:max_items]
        if not items:
            return None

        ny, nx = grid.ny, grid.nx
        g = grid.georef
        # north-up target transform in the grid CRS (origin = NW corner)
        dst_t = Affine(grid.dx, 0, g.x0, 0, -grid.dy, g.y0 + ny * grid.dy)
        dst_crs = f"EPSG:{epsg}"

        def read_band(item, band):
            dst = np.full((ny, nx), np.nan, np.float32)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                              GDAL_HTTP_MULTIPLEX="YES"):
                with rasterio.open(item.assets[band].href) as src:
                    reproject(rasterio.band(src, 1), dst,
                              dst_transform=dst_t, dst_crs=dst_crs,
                              resampling=Resampling.bilinear, src_nodata=0,
                              dst_nodata=np.nan)
            return dst

        vv_stack, vh_stack = [], []
        for it in items:
            try:
                vv_stack.append(read_band(it, "vv"))
                vh_stack.append(read_band(it, "vh"))
            except Exception:
                continue
        if not vv_stack:
            return None

        with np.errstate(invalid="ignore"):
            vv = np.nanmedian(np.stack(vv_stack), axis=0)
            vh = np.nanmedian(np.stack(vh_stack), axis=0)
            vh_vv = np.where(vv > 0, vh / vv, np.nan)
            rvi = np.where((vv + vh) > 0, 4 * vh / (vv + vh), np.nan)

        # north-up reproject -> flip to row 0 = south (grid convention); fill gaps
        def finish(a):
            a = np.flipud(a).astype(np.float32)
            m = np.nanmedian(a)
            return np.where(np.isfinite(a), a, m)

        return {
            "vv": finish(vv), "vh": finish(vh),
            "vh_vv": finish(vh_vv), "rvi": finish(rvi),
            "n_items": len(vv_stack),
            "dates": f"{items[-1].datetime.date()} – {items[0].datetime.date()}",
        }
    except Exception:
        return None


def fused_load_real(grid_load, sar_vh_vv, canopy_cover):
    """Real (uncalibrated) SAR+LiDAR blend over Eglin for visualization:
    trust the LiDAR load where canopy is open, fade to a SAR-informed estimate
    under canopy. Magnitudes are anchored to the LiDAR product's mean (absolute
    calibration pending co-located destructive truth)."""
    w = np.clip(canopy_cover, 0, 1)
    # scale SAR ratio into the LiDAR load's range (z-score -> match mean/std)
    r = sar_vh_vv
    r_z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-9)
    sar_est = np.clip(grid_load.mean() + r_z * grid_load.std(), 0, None)
    return ((1 - w) * grid_load + w * sar_est).astype(np.float32)
