"""Fuel moisture (challenge P2 properties) from free global data.

- DEAD fuel moisture: fine dead fuels equilibrate to the atmosphere, so we use the standard
  Simard (1968) equilibrium-moisture-content (EMC) model driven by ERA5 temperature + relative
  humidity (Copernicus CDS, needs ~/.cdsapirc). A defensible, weather-driven dead-FM estimate.
- LIVE fuel moisture: LFMC is hard from space; we provide a documented NDVI-scaled proxy from
  Sentinel-2 (Planetary Computer) — greener → wetter — bounded to a physical LFMC range.

Both are coarse relative to our 1 m grid (ERA5 ~9–31 km; LFMC a proxy), so they are reported as
near-uniform / smoothly-varying AOI fields, honestly flagged. Absolute calibration pending.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ── dead fuel moisture: Simard (1968) equilibrium moisture content ─────────────
def emc_simard(temp_C: float, rh_pct: float) -> float:
    """Equilibrium moisture content (%) of fine dead fuel from air temperature (°C) and
    relative humidity (%). Simard (1968), the standard NFDRS EMC (T in °F internally)."""
    T = temp_C * 9.0 / 5.0 + 32.0
    H = float(np.clip(rh_pct, 0.0, 100.0))
    if H < 10.0:
        emc = 0.03229 + 0.281073 * H - 0.000578 * H * T
    elif H <= 50.0:
        emc = 2.22749 + 0.160107 * H - 0.014784 * T
    else:
        emc = 21.0606 + 0.005565 * H * H - 0.00035 * H * T - 0.483199 * H
    return float(np.clip(emc, 1.0, 60.0))


def rh_from_dewpoint(temp_C: float, dewpoint_C: float) -> float:
    """Relative humidity (%) from temperature and dewpoint (°C), Magnus formula."""
    def g(t):
        return np.exp((17.625 * t) / (243.04 + t))
    return float(np.clip(100.0 * g(dewpoint_C) / g(temp_C), 1.0, 100.0))


# ── ERA5 fetch (Copernicus CDS) ────────────────────────────────────────────────
def era5_temp_rh(lat: float, lon: float, year: int, month: int = 7, day: int = 15,
                 hour: str = "18:00") -> Optional[Tuple[float, float]]:
    """Mean 2 m temperature (°C) + relative humidity (%) over a small box around (lat, lon)
    for one ERA5 hour (default mid-afternoon summer = near-peak fire weather). None on failure
    (e.g. the ERA5 licence not yet accepted at cds.climate.copernicus.eu)."""
    try:
        import cdsapi
        import tempfile
        import xarray as xr
        c = cdsapi.Client(quiet=True)
        out = os.path.join(tempfile.mkdtemp(), "era5.nc")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable": ["2m_temperature", "2m_dewpoint_temperature"],
            "year": str(year), "month": f"{month:02d}", "day": f"{day:02d}", "time": hour,
            "area": [lat + 0.15, lon - 0.15, lat - 0.15, lon + 0.15],  # N, W, S, E
            "format": "netcdf",
        }, out)
        ds = xr.open_dataset(out)
        t2 = float(ds["t2m"].mean()) - 273.15
        d2 = float(ds["d2m"].mean()) - 273.15
        return t2, rh_from_dewpoint(t2, d2)
    except Exception:
        return None


def dead_fuel_moisture(lat: float, lon: float, year: int) -> Optional[float]:
    """Fine dead fuel moisture (%) at the AOI = Simard EMC from ERA5 T + RH. None if ERA5 fails."""
    tr = era5_temp_rh(lat, lon, year)
    if tr is None:
        return None
    return emc_simard(tr[0], tr[1])


# ── live fuel moisture: NDVI-scaled proxy (Sentinel-2, Planetary Computer) ──────
def live_fuel_moisture_grid(grid, start="2023-05-01", end="2023-09-30",
                            lfmc_range=(60.0, 200.0)) -> Optional[np.ndarray]:
    """Coarse LFMC (%) proxy from a Sentinel-2 growing-season NDVI composite over the grid,
    linearly mapped NDVI[0.1,0.85] -> LFMC range. Spatially varying, honestly a proxy. row0=south."""
    try:
        import planetary_computer as pc
        import rasterio
        from rasterio.warp import reproject, Resampling
        from affine import Affine
        from pystac_client import Client
        from pyproj import Transformer
        g = grid.georef
        epsg = int(str(g.crs).split(":")[-1])
        tr = Transformer.from_crs(epsg, 4326, always_xy=True)
        xs, ys = tr.transform([g.x0, g.x0 + grid.nx * grid.dx, g.x0, g.x0 + grid.nx * grid.dx],
                              [g.y0, g.y0, g.y0 + grid.ny * grid.dy, g.y0 + grid.ny * grid.dy])
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        cat = Client.open(MPC_STAC, modifier=pc.sign_inplace)
        items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                datetime=f"{start}/{end}",
                                query={"eo:cloud_cover": {"lt": 20}}).items())[:6]
        if not items:
            return None
        ny, nx = grid.ny, grid.nx
        dst_t = Affine(grid.dx, 0, g.x0, 0, -grid.dy, g.y0 + ny * grid.dy)

        def band(it, key):
            out = np.full((ny, nx), np.nan, np.float32)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES"):
                with rasterio.open(it.assets[key].href) as src:
                    reproject(rasterio.band(src, 1), out, dst_transform=dst_t, dst_crs=f"EPSG:{epsg}",
                              resampling=Resampling.bilinear)
            return out
        ndvis = []
        for it in items:
            try:
                red, nir = band(it, "B04"), band(it, "B08")
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndvis.append((nir - red) / (nir + red))
            except Exception:
                continue
        if not ndvis:
            return None
        ndvi = np.nanmedian(np.stack(ndvis), 0)
        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nanmedian(ndvi))
        f = np.clip((ndvi - 0.1) / (0.85 - 0.1), 0, 1)
        lfmc = lfmc_range[0] + f * (lfmc_range[1] - lfmc_range[0])
        return np.flipud(lfmc).astype(np.float32)   # row0=south
    except Exception:
        return None
