"""Global coverage + vegetation-cover table for the v3 model. For a worldwide set of probe AOIs
spanning continents and biomes, report:
  COVERAGE   frac in-distribution (1 - frac_OOD) + verdict  -> where the model is grounded
  VEG COVER  ESA WorldCover dominant class + mean canopy height (m)  -> what's actually there
  STRUCTURE  v3 predicted understory occupancy (0-1)

Run:  python scripts/global_coverage_table.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import portable  # noqa: E402

WC_NAME = {10: "Tree", 20: "Shrub", 30: "Grassland", 40: "Cropland", 50: "Built",
           60: "Bare/sparse", 70: "Snow/ice", 80: "Water", 90: "Herb.wetland",
           95: "Mangrove", 100: "Moss/lichen", 0: "—"}

# (region, lat, lon, biome) — global spread across continents + biomes
REGIONS = [
    ("Florida (OSBS)*",     29.68,  -82.00, "subtrop. savanna"),
    ("Mojave (US)*",        35.90, -115.12, "desert scrub"),
    ("PNW (WREF)*",         45.82, -121.95, "temperate conifer"),
    ("Canada",              54.00,  -98.00, "boreal"),
    ("Amazon (Brazil)",     -4.50,  -62.50, "tropical rainforest"),
    ("Cerrado (Brazil)",   -13.00,  -47.00, "tropical savanna"),
    ("Germany*",            51.75,   10.60, "temperate broadleaf"),
    ("Sweden",              60.20,   15.60, "boreal"),
    ("Spain",               40.50,   -4.00, "Mediterranean"),
    ("Latvia (boreal)*",    56.91,   26.50, "hemiboreal conifer"),
    ("Congo basin",          0.50,   23.50, "tropical rainforest"),
    ("Sahara (Niger)",      18.00,   10.00, "hyper-arid desert"),
    ("Sahel (Mali)",        14.50,   -4.00, "arid grassland"),
    ("S.Africa savanna",   -24.00,   31.50, "savanna"),
    ("Borneo (SE Asia)",     1.20,  114.00, "tropical rainforest"),
    ("India (Deccan)",      18.00,   77.00, "dry tropical"),
    ("Australia outback",  -25.00,  133.00, "arid shrub"),
    ("E.Australia forest",  -33.00,  151.00, "temperate forest"),
]


def worldcover_class(grid):
    """Dominant ESA WorldCover class over the AOI (mode of valid pixels)."""
    try:
        import planetary_computer as pc
        import rasterio
        from rasterio.warp import reproject, Resampling
        from affine import Affine
        from pystac_client import Client
        g = grid.georef
        epsg = int(str(g.crs).split(":")[-1])
        from pyproj import Transformer
        tr = Transformer.from_crs(epsg, 4326, always_xy=True)
        xs, ys = tr.transform([g.x0, g.x0 + grid.nx * grid.dx, g.x0, g.x0 + grid.nx * grid.dx],
                              [g.y0, g.y0, g.y0 + grid.ny * grid.dy, g.y0 + grid.ny * grid.dy])
        cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        items = list(cat.search(collections=["esa-worldcover"],
                                bbox=[min(xs), min(ys), max(xs), max(ys)]).items())
        if not items:
            return 0
        items.sort(key=lambda it: it.properties.get("start_datetime", ""), reverse=True)
        ny, nx = grid.ny, grid.nx
        cls = np.zeros((ny, nx), np.uint8)
        dst_t = Affine(grid.dx, 0, g.x0, 0, -grid.dy, g.y0 + ny * grid.dy)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES"):
            with rasterio.open(items[0].assets["map"].href) as src:
                reproject(rasterio.band(src, 1), cls, dst_transform=dst_t, dst_crs=f"EPSG:{epsg}",
                          resampling=Resampling.nearest)
        vals, cnts = np.unique(cls[cls > 0], return_counts=True)
        return int(vals[cnts.argmax()]) if len(vals) else 0
    except Exception:
        return 0


def main():
    print(f"{'region':<20}{'biome':<22}{'WorldCover':<13}{'canopy m':>9}{'occ':>6}"
          f"{'in-dist%':>10}  coverage")
    print("-" * 92)
    for name, lat, lon, biome in REGIONS:
        try:
            r = portable.predict_aoi(lat, lon, size=300.0, year=2022, res=10, use_cache=True, version="v3")
            grid, _ = portable._aoi_grid(lat, lon, 300.0, 10.0)
            chm = portable.chm_for_grid(grid)
            wc = worldcover_class(grid)
            indist = (1.0 - r["frac_ood"]) * 100
            verdict = "✓ covered" if indist >= 50 else ("~ partial" if indist >= 10 else "✗ OOD")
            chm_m = float(np.nanmean(chm)) if chm is not None else float("nan")
            print(f"{name:<20}{biome:<22}{WC_NAME.get(wc,'?'):<13}{chm_m:>9.1f}"
                  f"{float(r['pred10'].mean()):>6.2f}{indist:>9.0f}%  {verdict}", flush=True)
        except Exception as e:
            print(f"{name:<20}{biome:<22}  failed: {type(e).__name__} {str(e)[:40]}", flush=True)
    print("\n* = a training site (or its biome). occ = predicted understory vertical occupancy (0-1).")
    print("in-dist% = 100·(1−frac_OOD): where ≥50% the v3 product is grounded; <10% is extrapolation.")


if __name__ == "__main__":
    main()
