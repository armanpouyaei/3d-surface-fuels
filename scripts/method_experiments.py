"""Can a pure-METHOD trick (no new data) lift WITHIN-biome transfer on the winning target
(occ_thin)? LOSO decomposition for:
  M0  baseline      GBM on raw AEF+S1 (point features)
  M1  +spatial      add 3x3 neighbourhood mean+std of every AEF channel (local texture)
  M2  2-stage resid stage-A predicts occ from AEF; stage-B predicts (occ - Â) from AEF+S1+context
If WITHIN R² stays negative for all, the within-biome residual is information-limited (needs
physical sub-canopy sensing: L-band SAR, canopy height, GEDI) — not a modelling problem.

Run:  python scripts/method_experiments.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from surface_fuels import metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TGT = "occ_thin"


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        s = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        g = dict(np.load(f))
        if TGT in g:
            out[s] = g
    return out


def gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6,
                                         learning_rate=0.08, max_iter=400, l2_regularization=1.0)


def base_feats(g):
    return [np.nan_to_num(g["aef"][k]) for k in range(64)] + \
           [np.nan_to_num(g["s1"][0]), np.nan_to_num(g["s1"][1])]


def spatial_feats(g):
    """3x3 neighbourhood mean+std of every AEF channel (local texture)."""
    extra = []
    for k in range(64):
        a = np.nan_to_num(g["aef"][k]).astype(np.float32)
        m = uniform_filter(a, 3)
        m2 = uniform_filter(a * a, 3)
        extra.append(m); extra.append(np.sqrt(np.clip(m2 - m * m, 0, None)))
    return extra


def stack(layers, valid):
    return np.stack([L.ravel() for L in layers], 1).astype(np.float32)[valid]


def decomp(P, Y, per_site_r2, pm, tm):
    return {"global_r2": M.r2(np.concatenate(P), np.concatenate(Y)),
            "within_r2": float(np.mean(per_site_r2)),
            "between_r2": M.r2(np.array(pm), np.array(tm))}


def run(grids, feat_fn):
    sites = list(grids)
    F = {s: stack(feat_fn(grids[s]), grids[s]["valid"].ravel()) for s in sites}
    y = {s: grids[s][TGT].ravel()[grids[s]["valid"].ravel()].astype(np.float32) for s in sites}
    P, Y, r2s, pm, tm = [], [], [], [], []
    for h in sites:
        Xtr = np.vstack([F[s] for s in sites if s != h]); ytr = np.concatenate([y[s] for s in sites if s != h])
        pred = np.clip(gbm().fit(Xtr, ytr).predict(F[h]), 0, None)
        P.append(pred); Y.append(y[h]); r2s.append(M.r2(pred, y[h])); pm.append(pred.mean()); tm.append(y[h].mean())
    return decomp(P, Y, r2s, pm, tm)


def run_2stage(grids):
    """Stage A: occ from AEF only. Stage B: residual from AEF+S1+3x3 context."""
    sites = list(grids)
    A = {s: stack([np.nan_to_num(grids[s]["aef"][k]) for k in range(64)], grids[s]["valid"].ravel()) for s in sites}
    B = {s: stack(base_feats(grids[s]) + spatial_feats(grids[s]), grids[s]["valid"].ravel()) for s in sites}
    y = {s: grids[s][TGT].ravel()[grids[s]["valid"].ravel()].astype(np.float32) for s in sites}
    P, Y, r2s, pm, tm = [], [], [], [], []
    for h in sites:
        oth = [s for s in sites if s != h]
        a_tr = np.vstack([A[s] for s in oth]); y_tr = np.concatenate([y[s] for s in oth])
        mA = gbm().fit(a_tr, y_tr)
        resid = np.concatenate([y[s] - mA.predict(A[s]) for s in oth])
        b_tr = np.vstack([B[s] for s in oth])
        mB = gbm().fit(b_tr, resid)
        pred = np.clip(mA.predict(A[h]) + mB.predict(B[h]), 0, None)
        P.append(pred); Y.append(y[h]); r2s.append(M.r2(pred, y[h])); pm.append(pred.mean()); tm.append(y[h].mean())
    return decomp(P, Y, r2s, pm, tm)


def main():
    grids = load_grids()
    print(f"target={TGT}, sites={list(grids)}\n")
    print(f"{'method':<18}{'GLOBAL R²':>11}{'WITHIN R²':>11}{'BETWEEN R²':>12}")
    print("-" * 52)
    for name, fn in [("M0 baseline", base_feats),
                     ("M1 +spatial 3x3", lambda g: base_feats(g) + spatial_feats(g))]:
        r = run(grids, fn)
        print(f"{name:<18}{r['global_r2']:>+11.3f}{r['within_r2']:>+11.3f}{r['between_r2']:>+12.3f}", flush=True)
    r = run_2stage(grids)
    print(f"{'M2 2-stage resid':<18}{r['global_r2']:>+11.3f}{r['within_r2']:>+11.3f}{r['between_r2']:>+12.3f}", flush=True)
    print("\nIf WITHIN R² stays <0 everywhere, the within-biome residual is INFORMATION-limited:")
    print("the next lever is physical sub-canopy data (L-band SAR / canopy height / GEDI), not method.")


if __name__ == "__main__":
    main()
