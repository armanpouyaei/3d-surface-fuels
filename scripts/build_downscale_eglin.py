"""Real 30 m -> 1 m downscaler over Eglin (Week 3).

Coarsens the measured 1 m LiDAR fuel grid to a ~30 m baseline, then reconstructs
1 m from globally-available covariates (real Sentinel-1 features, LiDAR canopy
cover, terrain) — validated by blocked CV. Saves data/processed/eglin_downscale.npz
for the dashboard.

Caveat: the Sentinel-1 here is 2026 while the LiDAR is 2007 (temporal gap), so
SAR contributes less than in a co-registered setting; canopy + terrain carry the
within-block signal. The synthetic demo (run_downscale_demo.py) is the clean
method validation.

Usage:  python scripts/build_downscale_eglin.py   (needs build_eglin + build_sar_eglin first)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, lidar, downscale as ds  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
LAZ = os.path.join(ROOT, "data", "raw", "eglin_3dep_000051.laz")
FACTOR = 30


def terrain_features(grid):
    """Elevation anomaly + slope over the AOI from the LiDAR ground DTM."""
    pc = lidar.load_points(LAZ, src_epsg=2238, target_epsg=32616, z_unit="ft_us")
    x0, y0 = grid.georef.x0, grid.georef.y0
    size = grid.nx * grid.dx
    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, size, res=2.0)
    elev = ds.upsample(dtm, int(round(size / ng / grid.dx)) or 1, (grid.ny, grid.nx))
    elev = elev[:grid.ny, :grid.nx] if elev.shape >= (grid.ny, grid.nx) else \
        np.pad(elev, ((0, grid.ny - elev.shape[0]), (0, grid.nx - elev.shape[1])), mode="edge")
    gy, gx = np.gradient(elev)
    slope = np.hypot(gy, gx)
    return (elev - elev.mean()).astype(np.float32), slope.astype(np.float32)


def main():
    mpath = os.path.join(PROC, "eglin_measured.nc")
    spath = os.path.join(PROC, "eglin_sar.npz")
    if not os.path.exists(mpath):
        sys.exit("Missing eglin_measured.nc — run scripts/build_eglin.py first.")
    grid = FuelVoxelGrid.from_netcdf(mpath)
    truth = grid.fuel_load()
    coarse = ds.block_coarsen(truth, FACTOR)
    print(f"Measured 1 m grid {truth.shape}; coarse baseline {coarse.shape} (~{FACTOR} m)")

    covars = {}
    if os.path.exists(spath):
        s = np.load(spath, allow_pickle=True)
        covars["s1_vhvv"] = s["vh_vv"].astype(np.float32)
        covars["s1_rvi"] = s["rvi"].astype(np.float32)
        covars["canopy"] = s["canopy"].astype(np.float32)
        print(f"  + Sentinel-1 features + canopy from eglin_sar.npz")
    else:
        covars["canopy"] = lidar.canopy_cover_grid(
            lidar.load_points(LAZ, 2238, 32616, "ft_us"), grid)
        print("  + canopy (no SAR npz — run build_sar_eglin.py for S1 features)")
    elev, slope = terrain_features(grid)
    covars["elev"], covars["slope"] = elev, slope
    print("  + terrain (elevation anomaly, slope)")

    out = ds.downscale_cv(truth, coarse, covars, FACTOR)
    rep = ds.evaluate(out, truth, FACTOR)
    b, d = rep["coarse_baseline"], rep["downscaler"]
    print(f"\n{'method':<22}{'R2':>8}{'within-block R2':>18}{'agg-consist':>13}")
    print(f"{'coarse upsample':<22}{b['r2']:>8}{b['within_block_r2']:>18}{b['agg_consistency_r2']:>13}")
    print(f"{'downscaler':<22}{d['r2']:>8}{d['within_block_r2']:>18}{d['agg_consistency_r2']:>13}")
    if out["feature_importance"] is not None:
        imp = sorted(zip(out["feature_names"], out["feature_importance"]), key=lambda t: -t[1])
        print("Feature importance:", {n: round(float(v), 2) for n, v in imp[:5]})

    np.savez_compressed(os.path.join(PROC, "eglin_downscale.npz"),
                        truth=truth, coarse=out["coarse_baseline"], downscaled=out["downscaled"],
                        factor=FACTOR)
    print("\nWrote data/processed/eglin_downscale.npz")


if __name__ == "__main__":
    main()
