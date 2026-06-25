"""Honest within-biome transfer for an UNSEEN biome, robust to low-variance sites.
WITHIN R² blows up where a site's own variance is tiny (e.g. shortgrass cper) — so we
report per-site WITHIN Spearman (rank, scale-free) and median WITHIN R², for base vs +L-band
on the occ_thin target. Question: does L-band SAR raise within-biome PATTERN transfer?

Run:  python scripts/eval_within_robust.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
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
        if TGT in g and "pal_hv" in g:
            out[s] = g
    return out


def feats(g, lband):
    L = [np.nan_to_num(g["aef"][k]) for k in range(64)] + \
        [np.nan_to_num(g["s1"][0]), np.nan_to_num(g["s1"][1])]
    if lband:
        L += [np.nan_to_num(g["pal_hh"]), np.nan_to_num(g["pal_hv"]), np.nan_to_num(g["pal_ratio"])]
    return np.stack([a.ravel() for a in L], 1).astype(np.float32)[g["valid"].ravel()]


def gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6,
                                         learning_rate=0.08, max_iter=400, l2_regularization=1.0)


def run(grids, lband):
    sites = list(grids)
    F = {s: feats(grids[s], lband) for s in sites}
    y = {s: grids[s][TGT].ravel()[grids[s]["valid"].ravel()].astype(np.float32) for s in sites}
    persp, perr2 = {}, {}
    for h in sites:
        Xtr = np.vstack([F[s] for s in sites if s != h]); ytr = np.concatenate([y[s] for s in sites if s != h])
        pred = np.clip(gbm().fit(Xtr, ytr).predict(F[h]), 0, None)
        persp[h] = spearmanr(pred, y[h]).correlation
        perr2[h] = M.r2(pred, y[h])
    return persp, perr2


def main():
    grids = load_grids()
    sites = list(grids)
    print(f"target={TGT}, sites={sites}\n")
    res = {}
    for lab, lb in [("base", False), ("+Lband", True)]:
        res[lab] = run(grids, lb)
    print("WITHIN Spearman (per-site rank transfer to an UNSEEN biome):")
    print(f"  {'site':<6}{'base':>8}{'+Lband':>9}{'Δ':>8}")
    for s in sites:
        b = res["base"][0][s]; l = res["+Lband"][0][s]
        print(f"  {s:<6}{b:>+8.2f}{l:>+9.2f}{l-b:>+8.2f}")
    bm = np.mean([res["base"][0][s] for s in sites]); lm = np.mean([res["+Lband"][0][s] for s in sites])
    print(f"  {'MEAN':<6}{bm:>+8.2f}{lm:>+9.2f}{lm-bm:>+8.2f}")
    bmed = np.median([res["base"][1][s] for s in sites]); lmed = np.median([res["+Lband"][1][s] for s in sites])
    npos_b = sum(res["base"][1][s] > 0 for s in sites); npos_l = sum(res["+Lband"][1][s] > 0 for s in sites)
    print(f"\nMEDIAN within R²:  base {bmed:+.2f}   +Lband {lmed:+.2f}")
    print(f"sites with within R²>0:  base {npos_b}/{len(sites)}   +Lband {npos_l}/{len(sites)}")


if __name__ == "__main__":
    main()
