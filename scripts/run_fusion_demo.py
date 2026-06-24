"""Synthetic SAR+LiDAR fusion validation (Week 2): spatially-blocked CV showing
fusion beats LiDAR-only, SAR-only, and the FastFuels uniform baseline — especially
under canopy, where LiDAR is occluded and SAR fills in.

Usage:  python scripts/run_fusion_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surface_fuels import fusion  # noqa: E402
from surface_fuels.synthetic import make_fusion_scenario  # noqa: E402


def main():
    sc = make_fusion_scenario(nx=120, ny=120, seed=42)
    print(f"Scene: 120x120, canopy cover mean {sc.canopy.mean():.2f}, "
          f"under-canopy (>0.3) {100*(sc.canopy>=0.3).mean():.0f}%\n")
    rep = fusion.evaluate(fusion.spatial_block_cv(sc, blocks=4), sc)

    hdr = f"{'method':<20}{'R2':>8}{'RMSE':>9}{'R2 open':>10}{'R2 canopy':>11}"
    print(hdr); print("-" * len(hdr))
    for k in ["lidar_only", "sar_only", "fusion", "fastfuels_uniform"]:
        r = rep[k]
        print(f"{k:<20}{r['r2']:>8}{r['rmse']:>9}{r['r2_open']:>10}{r['r2_under_canopy']:>11}")

    f, l = rep["fusion"], rep["lidar_only"]
    print(f"\nFusion improves overall R2 by {f['r2']-l['r2']:+.2f} vs LiDAR-only, "
          f"and under canopy by {f['r2_under_canopy']-l['r2_under_canopy']:+.2f} "
          f"(LiDAR is occluded there). Radar fills LiDAR's under-canopy blind spot.")


if __name__ == "__main__":
    main()
