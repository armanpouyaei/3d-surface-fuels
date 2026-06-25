"""Final feature ablation on the full site set (occ_thin target, LOSO). Does canopy height (b)
lift within-forest transfer beyond L-band? Reports both transfer axes + a robust within metric.

feature sets:  base = AEF+S1 ; +L = +PALSAR ; +L+chm = +canopy height ; +L+chm+terr = +terrain
metrics:  GLOBAL R² (absolute level) · BETWEEN R² (biome ranking) · mean Spearman ·
          MEDIAN within R² (robust to near-uniform sites) · #sites within R²>0 ·
          forest within-Spearman mean (the (b) target — unseen-forest fine structure)

Run:  python scripts/eval_features_final.py
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
FORESTS = {"osbs", "soap", "wref", "harv", "fr", "ch", "nl", "pr", "lva"}   # have real understory


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        s = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        g = dict(np.load(f))
        if TGT in g and "pal_hv" in g and "chm" in g:
            out[s] = g
    return out


def layers(g, kind):
    aef = [np.nan_to_num(g["aef"][k]) for k in range(64)]
    s1 = [np.nan_to_num(g["s1"][0]), np.nan_to_num(g["s1"][1])]
    lb = [np.nan_to_num(g["pal_hh"]), np.nan_to_num(g["pal_hv"]), np.nan_to_num(g["pal_ratio"])]
    chm = [np.nan_to_num(g["chm"])]
    tr = [np.nan_to_num(g["elev"]), np.nan_to_num(g["slope"]), np.nan_to_num(g["tpi"])]
    return {"base": aef + s1, "+L": aef + s1 + lb, "+L+chm": aef + s1 + lb + chm,
            "+L+chm+terr": aef + s1 + lb + chm + tr}[kind]


def gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6,
                                         learning_rate=0.08, max_iter=400, l2_regularization=1.0)


def loso(grids, kind):
    sites = list(grids)
    F = {s: np.stack([a.ravel() for a in layers(grids[s], kind)], 1).astype(np.float32)[grids[s]["valid"].ravel()]
         for s in sites}
    y = {s: grids[s][TGT].ravel()[grids[s]["valid"].ravel()].astype(np.float32) for s in sites}
    P, Y, r2s, sps, pm, tm = [], [], [], {}, [], []
    for h in sites:
        Xtr = np.vstack([F[s] for s in sites if s != h]); ytr = np.concatenate([y[s] for s in sites if s != h])
        pred = np.clip(gbm().fit(Xtr, ytr).predict(F[h]), 0, None)
        P.append(pred); Y.append(y[h]); r2s.append(M.r2(pred, y[h]))
        sps[h] = spearmanr(pred, y[h]).correlation; pm.append(pred.mean()); tm.append(y[h].mean())
    r2arr = np.array(r2s)
    fore = [sps[s] for s in sites if s in FORESTS]
    return {"global": M.r2(np.concatenate(P), np.concatenate(Y)), "between": M.r2(np.array(pm), np.array(tm)),
            "spear": float(np.mean(list(sps.values()))), "med_within": float(np.median(r2arr)),
            "npos": int((r2arr > 0).sum()), "n": len(sites), "forest_sp": float(np.mean(fore))}


def main():
    grids = load_grids()
    print(f"target={TGT}  sites({len(grids)})={list(grids)}\n")
    print(f"{'features':<14}{'GLOBAL':>8}{'BETWEEN':>9}{'Spear':>7}{'medWithin':>11}{'#>0':>6}{'forestSp':>10}")
    print("-" * 65)
    for kind in ["base", "+L", "+L+chm", "+L+chm+terr"]:
        r = loso(grids, kind)
        print(f"{kind:<14}{r['global']:>+8.3f}{r['between']:>+9.3f}{r['spear']:>+7.3f}"
              f"{r['med_within']:>+11.3f}{r['npos']:>4}/{r['n']:<1}{r['forest_sp']:>+10.3f}", flush=True)
    print("\nforestSp = mean within-site Spearman over forested sites (the (b) target).")
    print("If +chm raises forestSp / medWithin, canopy height (GEDI-densified) adds under-canopy info.")


if __name__ == "__main__":
    main()
