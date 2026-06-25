"""Train portable model **v2** from cached per-site grids (fast — no re-fetch).

v2 = quantile GBM on **per-tile-standardized AlphaEarth** (+ raw Sentinel-1) — domain
adaptation that lets the structure relationship transfer across ecosystems (validated by
the leave-one-site-out experiments: mean R² −0.62→~0, Spearman +0.17→+0.30, and OSBS
flips from −0.36 to +0.18). OOD is kept on RAW features so domain shift is still flagged.

Usage:  python scripts/train_portable_v2.py   (needs data/interim/portable_grid_*.npz)
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import portable  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    grids = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        site = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = np.load(f)
        grids[site] = {k: d[k] for k in d.files}
    if not grids:
        sys.exit("No cached grids — run scripts/train_portable.py first.")
    sites = list(grids)
    path = portable.train_v2(grids, sites)
    n = sum(int(grids[s]["valid"].sum()) for s in sites)
    print(f"Saved v2 (tile-std domain-adapted) on {sites} ({n} samples) → {path}")
    print("Inference: portable.predict_aoi(lat, lon, version='v2').")


if __name__ == "__main__":
    main()
