"""v2 generalization experiments — what makes the portable model transfer to UNSEEN
ecosystems? Reads the cached per-site grids (data/interim/portable_grid_*.npz) and runs
leave-one-site-out (LOSO) for several treatments, reporting R² (absolute skill) AND
Spearman (does the spatial PATTERN transfer?). Cheap GBM variants first to isolate the
generalization levers, then (optionally) the spatial CNN.

Treatments:
  v1            — per-pixel GBM on raw AEF+S1 (the current model)
  +tile-std     — standardize AEF per-tile (domain adaptation: removes per-region offset/scale)
  +tile-std+rank— predict within-tile RANK (0-1) of structure (scale-free pattern target)

Usage:  python scripts/portable_v2_experiments.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from surface_fuels import metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        site = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = np.load(f)
        out[site] = {k: d[k] for k in d.files}
    return out


def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6, learning_rate=0.08,
                                         max_iter=200, l2_regularization=1.0, random_state=0)


def site_features(g, tile_std):
    """Flat (N,66) features + target + within-tile rank, for valid cells."""
    aef = g["aef"].astype(np.float32)               # (64,ny,nx)
    if tile_std:
        m = aef.reshape(64, -1).mean(1)[:, None, None]
        s = aef.reshape(64, -1).std(1)[:, None, None] + 1e-6
        aef = (aef - m) / s
    s1 = g["s1"].astype(np.float32)                  # (2,ny,nx)
    stack = np.concatenate([aef, s1], 0)             # (66,ny,nx)
    valid = g["valid"].ravel()
    X = stack.reshape(stack.shape[0], -1).T[valid]
    y = g["target"].ravel()[valid]
    r = rankdata(y) / max(len(y), 1)                 # within-tile rank 0-1
    return X.astype(np.float32), y.astype(np.float32), r.astype(np.float32)


def loso(data, tile_std, rank_target):
    names = list(data)
    feats = {n: site_features(data[n], tile_std) for n in names}
    r2s, sps = [], []
    rows = []
    for held in names:
        Xtr = np.vstack([feats[n][0] for n in names if n != held])
        ytr = np.concatenate([feats[n][2 if rank_target else 1] for n in names if n != held])
        Xte, yte_abs, yte_rank = feats[held]
        pred = np.clip(gbm().fit(Xtr, ytr).predict(Xte), 0, None)
        ref = yte_rank if rank_target else yte_abs
        r2 = M.r2(pred, ref); sp = spearmanr(pred, yte_abs).correlation
        r2s.append(r2); sps.append(sp); rows.append((held, r2, sp))
    return r2s, sps, rows


def main():
    data = load_grids()
    if len(data) < 2:
        sys.exit("Need cached grids — run scripts/train_portable.py first (it caches data/interim/portable_grid_*.npz).")
    print(f"Loaded grids: {list(data)}  (shapes: " +
          ", ".join(f"{n}{tuple(data[n]['target'].shape)}" for n in data) + ")\n")
    treatments = [("v1 (raw AEF+S1)", False, False),
                  ("+tile-std (domain-adapt)", True, False),
                  ("+tile-std +rank target", True, True)]
    print(f"{'treatment':<28}{'mean R²':>9}{'mean Spearman':>15}   per-site Spearman")
    best = None
    for label, tstd, rank in treatments:
        r2s, sps, rows = loso(data, tstd, rank)
        persite = " ".join(f"{h}:{sp:+.2f}" for h, _, sp in rows)
        print(f"{label:<28}{np.mean(r2s):>+9.3f}{np.mean(sps):>+15.3f}   {persite}")
        if best is None or np.mean(sps) > best[1]:
            best = (label, float(np.mean(sps)), float(np.mean(r2s)))
    print(f"\nBest by mean Spearman (pattern transfer): {best[0]}  (Spearman {best[1]:+.3f}, R² {best[2]:+.3f})")
    print("Spearman>0 ⇒ the spatial PATTERN transfers cross-ecosystem (scale via per-AOI calibration);")
    print("if tile-std lifts Spearman, domain adaptation is the generalization win to bake into v2.")


if __name__ == "__main__":
    main()
