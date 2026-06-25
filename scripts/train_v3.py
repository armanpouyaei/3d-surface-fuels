"""Train the v3 portable model — the cracked cross-ecosystem config (RESULTS.md §2f):
  target   = vertical occupancy occ_thin (density-robust structure metric)
  features = RAW AlphaEarth + Sentinel-1 + L-band PALSAR HH/HV/ratio  (NO domain adaptation)
LOSO: GLOBAL R² 0.46, BETWEEN-biome 0.76. Saves data/processed/stage1_portable_v3.joblib.

Requires the cached grids to already carry occ_thin (rederive_thinned.py) and pal_* features
(add_physical_features.py). Run:  python scripts/train_v3.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from surface_fuels import portable, metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        s = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        g = dict(np.load(f))
        if "occ_thin" in g and "pal_hv" in g and "chm" in g:
            out[s] = g
    return out


def main():
    grids = load_grids()
    sites = list(grids)
    if not sites:
        sys.exit("No grids with occ_thin + L-band — run rederive_thinned.py & add_physical_features.py.")
    path = portable.train_v3(grids, sites)
    m = portable.load("v3")
    print(f"Saved v3 ({m['n_train']} samples, {len(sites)} sites) → {path}")
    print(f"  target={m['target']}  use_lband={m['use_lband']}  use_chm={m.get('use_chm')}  "
          f"feats={len(m['feats'])}  ood_thresh={m['ood_thresh']:.2f}")
    print("  LOSO (eval_features_final.py, 13 sites): GLOBAL R² 0.49 / BETWEEN 0.81 / forest within-Spearman 0.44")


if __name__ == "__main__":
    main()
