"""Build the real Eglin surface-fuel demo from USGS 3DEP LiDAR.

Pipeline: LAZ point cloud -> 1 m surface-fuel bulk-density grid (measured) +
uniform FastFuels-style baseline + a real Esri World Imagery basemap for the
AOI. Outputs land in data/processed/ for the dashboard's "Real Eglin" mode.

Usage:  python scripts/build_eglin.py [--size 240] [--load 0.6]
Requires data/raw/eglin_3dep_000051.laz (see scripts/download_eglin_lidar.py).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import lidar, metrics  # noqa: E402
from surface_fuels.basemap import basemap_for_grid  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAZ = os.path.join(ROOT, "data", "raw", "eglin_3dep_000051.laz")
OUT = os.path.join(ROOT, "data", "processed")


def _canopy_height_grid(pc, grid, ng):
    """Coarse max-canopy-height grid over the AOI (row 0 = south), for the
    same-spot/orientation check against the imagery."""
    x0, y0 = grid.georef.x0, grid.georef.y0
    size = grid.nx * grid.dx
    dtm, dres, ngc = lidar._ground_dtm(pc, x0, y0, size, res=5.0)
    m = (pc.x >= x0) & (pc.x < x0 + size) & (pc.y >= y0) & (pc.y < y0 + size)
    px, py, pz = pc.x[m], pc.y[m], pc.z[m]
    gi = np.clip(((px - x0) / dres).astype(int), 0, ngc - 1)
    gj = np.clip(((py - y0) / dres).astype(int), 0, ngc - 1)
    h = np.clip(pz - dtm[gj, gi], 0, None)
    ci = np.clip(((px - x0) / (size / ng)).astype(int), 0, ng - 1)
    cj = np.clip(((py - y0) / (size / ng)).astype(int), 0, ng - 1)
    chm = np.zeros((ng, ng))
    np.maximum.at(chm, (cj, ci), h)
    return chm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=240.0, help="AOI side length (m)")
    ap.add_argument("--nz", type=int, default=4, help="surface stratum height (m)")
    ap.add_argument("--load", type=float, default=0.6, help="mean load anchor (kg/m^2)")
    ap.add_argument("--aoi", choices=["open", "center"], default="open",
                    help="'open': auto-pick an open-but-structured window; 'center': tile centroid")
    args = ap.parse_args()

    if not os.path.exists(LAZ):
        sys.exit(f"Missing {LAZ}. Run scripts/download_eglin_lidar.py first.")
    os.makedirs(OUT, exist_ok=True)

    print("Loading point cloud (reproject EPSG:2238 ftUS -> UTM 16N m)...")
    pc = lidar.load_points(LAZ, src_epsg=2238, target_epsg=32616, z_unit="ft_us")
    print(f"  {len(pc.x):,} points; ground returns: {(pc.cls == 2).sum():,}")

    center = None
    if args.aoi == "open":
        center, info = lidar.find_aoi(pc, size=args.size)
        print(f"Selected open AOI center UTM={tuple(round(c) for c in center)} "
              f"(canopy cover {info['mean_cover']}, structure {info['structure']})")
        from pyproj import Transformer
        tr = Transformer.from_crs(32616, 4326, always_xy=True)
        sw = tr.transform(center[0] - args.size / 2, center[1] - args.size / 2)
        ne = tr.transform(center[0] + args.size / 2, center[1] + args.size / 2)
        print(f"  AOI bbox lon/lat: ({sw[0]:.5f}, {sw[1]:.5f}) – ({ne[0]:.5f}, {ne[1]:.5f})  [Eglin AFB, FL]")

    print(f"Voxelizing {args.size:.0f} m AOI to 1 m x 1 m x 1 m, 0-{args.nz} m stratum...")
    grid = lidar.voxelize(pc, center=center, size=args.size, nz=args.nz, target_load_kg_m2=args.load)
    base = lidar.uniform_baseline(grid)

    grid.to_netcdf(os.path.join(OUT, "eglin_measured.nc"))
    base.to_netcdf(os.path.join(OUT, "eglin_uniform.nc"))
    print("  wrote eglin_measured.nc, eglin_uniform.nc")
    print("  measured summary:", json.dumps(grid.summary()))

    print("Fetching real Esri World Imagery basemap for the AOI...")
    # cache key tied to AOI so changing the window refetches the right tile
    bpath = os.path.join(OUT, "eglin_basemap.png")
    if os.path.exists(bpath):
        os.remove(bpath)
    rgb = basemap_for_grid(grid, px=600, cache_path=bpath)
    print("  basemap:", "saved eglin_basemap.png" if rgb is not None else "FETCH FAILED (offline?)")

    # Same-spot / orientation check using LiDAR canopy height (a strong feature):
    # tall vegetation should be darker + greener in the imagery. The basemap and
    # LiDAR share the row 0 = south orientation (no relative flip).
    if rgb is not None:
        from PIL import Image
        ng = 40
        chm = _canopy_height_grid(pc, grid, ng)            # row 0 = south
        small = np.asarray(Image.fromarray(rgb).resize((ng, ng))).astype(float)
        bright = small[..., :3].mean(axis=2)
        green = small[..., 1] - 0.5 * (small[..., 0] + small[..., 2])
        def corr(a, b):
            return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        print("\nSame-spot/orientation check (LiDAR canopy height vs imagery, same orientation):")
        print(f"  corr(canopy, brightness) = {corr(chm, bright):+.3f}  (taller veg darker -> expect negative)")
        print(f"  corr(canopy, greenness)  = {corr(chm, green):+.3f}  (taller veg greener -> expect positive)")

    # heterogeneity that the uniform baseline destroys
    het_m = metrics.heterogeneity_stats(grid)
    het_b = metrics.heterogeneity_stats(base)
    print("\nHeterogeneity (measured vs uniform baseline):")
    print(f"  measured  CV = {het_m['cv']:.3f}  Moran's I = {het_m['moran_i']:.3f}  mean load = {het_m['mean_load_kg_m2']:.3f} kg/m2")
    print(f"  uniform   CV = {het_b['cv']:.3f}  (a single value per fuel class)")
    occ = grid.occupancy(1e-6).mean()
    print(f"  occupied surface voxels: {occ * 100:.1f}%  (sparse = real ALS under-sampling of understory)")


if __name__ == "__main__":
    main()
