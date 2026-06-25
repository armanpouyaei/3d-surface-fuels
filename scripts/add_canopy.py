"""(b) Add Meta/WRI global canopy height (GEDI+ALS-calibrated, 1 m) as a `chm` feature to
each cached grid — the gridded form of GEDI vertical structure. Free/ungated AWS COG, fetched
per grid like S1/PALSAR. Run:  python scripts/add_canopy.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import portable, FuelVoxelGrid, GeoRef  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def grid_from(d):
    ny, nx = d["target"].shape
    return FuelVoxelGrid(np.zeros((1, ny, nx), np.float32), dz=float(d["res"]), dy=float(d["res"]),
                         dx=float(d["res"]),
                         georef=GeoRef(f"EPSG:{int(d['epsg'])}", float(d["x0"]), float(d["y0"]), float(d["res"])))


def main():
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        s = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = dict(np.load(f))
        if "chm" in d:
            print(f"  [{s}] chm present — skip", flush=True); continue
        chm = portable.chm_for_grid(grid_from(d))
        if chm is None:
            print(f"  [{s}] chm fetch failed — skip", flush=True); continue
        d["chm"] = chm.astype(np.float32)
        np.savez_compressed(f, **d)
        v = d["valid"]
        print(f"  [{s}] chm μ{chm[v].mean():.1f} m  p95 {np.percentile(chm[v],95):.1f} m", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
