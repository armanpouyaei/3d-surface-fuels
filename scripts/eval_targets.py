"""Which target + method recovers REAL cross-ecosystem skill? Leave-one-site-out on the
cached 10-site grids, for each candidate target, decomposed into the two transfer problems:

  GLOBAL R²   (vs grand mean)  -- "place an unseen biome at the right ABSOLUTE level"
                                  = the between-biome signal AEF can supply; what a global
                                  product needs. Killed by the old mean-0.6 normalization.
  WITHIN R²   (vs held-out site mean) -- "predict fine heterogeneity INSIDE an unseen biome"
                                  = the genuinely hard, sensing-limited residual (current metric).
  BETWEEN R²  (site-mean pred vs true, across folds) -- can we rank/scale the 10 biomes?
  density confound -- corr(site-mean target, site-mean LiDAR density); high = the between-biome
                                  variance may be a SENSOR artifact, not vegetation.

Methods: raw AEF+S1 GBM  vs  per-tile-standardized-AEF GBM (v2 domain adaptation). tile-std
removes the per-site offset -> should HELP within, HURT global (it erases biome level).

Run:  python scripts/eval_targets.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from surface_fuels import metrics as M, portable  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TARGETS = ["target", "frac_raw", "cover", "occ_vert", "pad_gap",
           "occ_thin", "cover_thin"]  # *_thin = density-equalized (sensor-artifact control)


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        s = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        out[s] = dict(np.load(f))
    return out


def feats(g, tile_std):
    v = g["valid"].ravel()
    X = np.stack([np.nan_to_num(g["aef"][k]).ravel() for k in range(64)]
                 + [np.nan_to_num(g["s1"][0]).ravel(), np.nan_to_num(g["s1"][1]).ravel()], 1)[v]
    if tile_std:
        X = portable._tile_std_aef(X, n_aef=64)
    return X.astype(np.float32)


def gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6,
                                         learning_rate=0.08, max_iter=400, l2_regularization=1.0)


def loso(grids, tgt, tile_std):
    sites = [s for s in grids if tgt in grids[s]]
    feat = {s: feats(grids[s], tile_std) for s in sites}
    y = {s: grids[s][tgt].ravel()[grids[s]["valid"].ravel()].astype(np.float32) for s in sites}
    P, Y, win_r2, sp, pm, tm = [], [], [], [], [], []
    for h in sites:
        Xtr = np.vstack([feat[s] for s in sites if s != h])
        ytr = np.concatenate([y[s] for s in sites if s != h])
        pred = np.clip(gbm().fit(Xtr, ytr).predict(feat[h]), 0, None)
        P.append(pred); Y.append(y[h])
        win_r2.append(M.r2(pred, y[h])); sp.append(spearmanr(pred, y[h]).correlation)
        pm.append(pred.mean()); tm.append(y[h].mean())
    P = np.concatenate(P); Y = np.concatenate(Y)
    return {"global_r2": M.r2(P, Y), "within_r2": float(np.mean(win_r2)),
            "spearman": float(np.mean(sp)), "between_r2": M.r2(np.array(pm), np.array(tm)),
            "sites": sites, "true_means": dict(zip(sites, tm))}


def density_confound(grids, tgt):
    sites = [s for s in grids if tgt in grids[s] and "dens" in grids[s]]
    tmean = [grids[s][tgt].ravel()[grids[s]["valid"].ravel()].mean() for s in sites]
    dmean = [grids[s]["dens"].ravel()[grids[s]["valid"].ravel()].mean() for s in sites]
    return spearmanr(tmean, dmean).correlation


def main():
    grids = load_grids()
    print(f"{'target':<10}{'method':<10}{'GLOBAL R²':>11}{'WITHIN R²':>11}{'BETWEEN R²':>12}"
          f"{'Spearman':>10}{'ρ(tgt,dens)':>13}")
    print("-" * 77)
    for tgt in TARGETS:
        if not any(tgt in grids[s] for s in grids):
            print(f"{tgt:<10}  (not derived yet — run rederive_targets.py)"); continue
        conf = density_confound(grids, tgt)
        for label, ts in [("raw", False), ("tile-std", True)]:
            r = loso(grids, tgt, ts)
            print(f"{tgt:<10}{label:<10}{r['global_r2']:>+11.3f}{r['within_r2']:>+11.3f}"
                  f"{r['between_r2']:>+12.3f}{r['spearman']:>+10.3f}{conf:>+13.2f}", flush=True)
    print("\nGLOBAL R² = can we put an unseen biome at the right absolute level (product-relevant).")
    print("WITHIN R² = fine heterogeneity inside an unseen biome (the hard, sensing-limited residual).")
    print("ρ(tgt,dens) near 0 = between-biome signal is vegetation, not a LiDAR-density artifact.")


if __name__ == "__main__":
    main()
