"""Fetch real Sentinel-1 SAR over the Eglin AOI and build a SAR+LiDAR fused
surface-fuel product (Week 2).

Pipeline: load the measured LiDAR grid (scripts/build_eglin.py output) -> compute
per-cell canopy cover from the point cloud -> fetch median Sentinel-1 RTC features
(VV/VH/ratio) from Planetary Computer over the same footprint -> blend (LiDAR in
the open, SAR under canopy). Saves data/processed/eglin_sar.npz for the dashboard.

Usage:  python scripts/build_sar_eglin.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, lidar, sar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
LAZ = os.path.join(ROOT, "data", "raw", "eglin_3dep_000051.laz")


def main():
    mpath = os.path.join(PROC, "eglin_measured.nc")
    if not os.path.exists(mpath):
        sys.exit("Missing eglin_measured.nc — run scripts/build_eglin.py first.")
    grid = FuelVoxelGrid.from_netcdf(mpath)
    load = grid.fuel_load()
    print(f"Measured grid {grid.shape}  mean load {load.mean():.2f} kg/m^2")

    print("Computing per-cell canopy cover from the point cloud...")
    pc = lidar.load_points(LAZ, src_epsg=2238, target_epsg=32616, z_unit="ft_us")
    canopy = lidar.canopy_cover_grid(pc, grid)
    print(f"  canopy cover: mean {canopy.mean():.2f}, under-canopy (>0.3) {100*(canopy>0.3).mean():.0f}%")

    print("Fetching real Sentinel-1 RTC (Planetary Computer)...")
    feats = sar.sar_features_for_grid(grid)
    if feats is None:
        sys.exit("Sentinel-1 fetch failed (offline?). SAR features unavailable.")
    print(f"  {feats['n_items']} acquisitions composited ({feats['dates']})")
    print(f"  VH/VV ratio: mean {np.nanmean(feats['vh_vv']):.3f}  RVI mean {np.nanmean(feats['rvi']):.3f}")

    fused = sar.fused_load_real(load, feats["vh_vv"], canopy)
    print(f"  fused load: mean {fused.mean():.2f} kg/m^2  CV {fused.std()/fused.mean():.2f}")

    np.savez_compressed(
        os.path.join(PROC, "eglin_sar.npz"),
        vv=feats["vv"], vh=feats["vh"], vh_vv=feats["vh_vv"], rvi=feats["rvi"],
        canopy=canopy, lidar_load=load, fused=fused,
        dates=feats["dates"], n_items=feats["n_items"],
    )
    print("Wrote data/processed/eglin_sar.npz")


if __name__ == "__main__":
    main()
