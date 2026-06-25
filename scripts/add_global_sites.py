"""Broaden training coverage: build grids for the new international + US-gap ALS sites,
then retrain BOTH v1 (raw) and v2 (per-tile domain-adapted) on ALL cached sites, and
report leave-one-site-out R²+Spearman for each. Reuses the 6 already-cached grids — only
the new sites are fetched.

New sites (open ALS, vintage-matched to AlphaEarth):
  ch    Switzerland swissSURFACE3D (temperate/alpine forest, EPSG 2056, 2019)
  nl    Netherlands AHN4 (heath + Scots pine, EPSG 28992, 2022)
  fr    France IGN LiDAR HD (montane mixed forest, EPSG 2154, ~2024)
  akf   USGS 3DEP Alaska Fairbanks (boreal taiga, EPSG 6335 ftUS, 2017)
  ever  USGS 3DEP Everglades (wetland/swamp, EPSG 6346, 2017)

Usage:  python scripts/add_global_sites.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))            # for sibling imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from surface_fuels import portable, metrics as M  # noqa: E402
from train_portable import site_samples  # noqa: E402  (builds + caches a site's grid)
from portable_v2_experiments import loso  # noqa: E402  (LOSO with optional tile-std)

ROOT = os.path.join(os.path.dirname(__file__), "..")
NEW = [
    {"name": "ch",   "laz": "data/raw/ch.las",   "src": 2056,  "tgt": 32632, "year": 2019, "z": "m"},
    {"name": "nl",   "laz": "data/raw/nl.laz",   "src": 28992, "tgt": 32631, "year": 2022, "z": "m"},
    {"name": "fr",   "laz": "data/raw/fr.laz",   "src": 2154,  "tgt": 32631, "year": 2024, "z": "m"},
    {"name": "akf",  "laz": "data/raw/akf.laz",  "src": 6335,  "tgt": 32606, "year": 2018, "z": "ft_us"},
    {"name": "ever", "laz": "data/raw/ever.laz", "src": 6346,  "tgt": 32617, "year": 2017, "z": "m"},
]


def load_all_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        site = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = np.load(f); out[site] = {k: d[k] for k in d.files}
    return out


def flat_raw(g):
    cols = [np.nan_to_num(g["aef"][i]).ravel() for i in range(64)]
    cols += [np.nan_to_num(g["s1"][0]).ravel(), np.nan_to_num(g["s1"][1]).ravel()]
    v = g["valid"].ravel()
    return np.stack(cols, 1).astype(np.float32)[v], g["target"].ravel()[v].astype(np.float32)


def main():
    for s in NEW:
        gp = os.path.join(ROOT, "data", "interim", f"portable_grid_{s['name']}.npz")
        if os.path.exists(gp):
            print(f"[{s['name']}] grid already cached — skip", flush=True); continue
        if not os.path.exists(os.path.join(ROOT, s["laz"])):
            print(f"[{s['name']}] LAZ missing ({s['laz']}) — skip", flush=True); continue
        try:
            site_samples(s)                       # builds + caches grid (AEF+S1 fetch)
        except Exception as e:
            print(f"[{s['name']}] build FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)

    grids = load_all_grids()
    sites = list(grids)
    print(f"\nAll cached sites ({len(sites)}): {sites}\n")

    print(f"{'treatment':<22}{'mean R²':>9}{'mean Spearman':>15}   per-site Spearman")
    for label, tstd in [("v1 raw", False), ("v2 tile-std", True)]:
        r2s, sps, rows = loso(grids, tstd, False)
        persite = " ".join(f"{h}:{sp:+.2f}" for h, _, sp in rows)
        print(f"{label:<22}{np.mean(r2s):>+9.3f}{np.mean(sps):>+15.3f}   {persite}", flush=True)

    # retrain + save both models on ALL sites
    Xs, ys = zip(*(flat_raw(grids[s]) for s in sites))
    portable.train(np.vstack(Xs), np.concatenate(ys), sites=sites)         # v1
    portable.train_v2(grids, sites)                                        # v2
    print(f"\nSaved v1 + v2 on {len(sites)} sites ({sum(len(y) for y in ys)} samples).")


if __name__ == "__main__":
    main()
