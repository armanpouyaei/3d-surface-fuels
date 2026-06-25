"""Do PHYSICAL sub-canopy features (L-band PALSAR + terrain) crack the within-biome residual?
LOSO decomposition on the winning target (occ_thin), feature-set ablation:
  base    AEF + Sentinel-1            (optical + C-band — canopy-top)
  +Lband  base + PALSAR HH/HV/ratio   (L-band — canopy-penetrating)
  +terr   base + elevation/slope/TPI
  +both   base + PALSAR + terrain
  phys    PALSAR + terrain only       (how much do the physical sensors carry alone?)

The headline question: does WITHIN R² (fine heterogeneity in an UNSEEN biome) move off the
floor (~ -1.2) when L-band is added? Run:  python scripts/eval_physical.py
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


def layers(g, kind):
    aef = [np.nan_to_num(g["aef"][k]) for k in range(64)]
    s1 = [np.nan_to_num(g["s1"][0]), np.nan_to_num(g["s1"][1])]
    lb = [np.nan_to_num(g["pal_hh"]), np.nan_to_num(g["pal_hv"]), np.nan_to_num(g["pal_ratio"])]
    tr = [np.nan_to_num(g["elev"]), np.nan_to_num(g["slope"]), np.nan_to_num(g["tpi"])]
    return {"base": aef + s1, "+Lband": aef + s1 + lb, "+terr": aef + s1 + tr,
            "+both": aef + s1 + lb + tr, "phys": lb + tr}[kind]


def gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6,
                                         learning_rate=0.08, max_iter=400, l2_regularization=1.0)


def loso(grids, kind):
    sites = list(grids)
    F = {s: np.stack([L.ravel() for L in layers(grids[s], kind)], 1).astype(np.float32)[grids[s]["valid"].ravel()]
         for s in sites}
    y = {s: grids[s][TGT].ravel()[grids[s]["valid"].ravel()].astype(np.float32) for s in sites}
    P, Y, r2s, sps, pm, tm = [], [], [], [], [], []
    for h in sites:
        Xtr = np.vstack([F[s] for s in sites if s != h]); ytr = np.concatenate([y[s] for s in sites if s != h])
        pred = np.clip(gbm().fit(Xtr, ytr).predict(F[h]), 0, None)
        P.append(pred); Y.append(y[h]); r2s.append(M.r2(pred, y[h]))
        sps.append(spearmanr(pred, y[h]).correlation); pm.append(pred.mean()); tm.append(y[h].mean())
    return {"global": M.r2(np.concatenate(P), np.concatenate(Y)), "within": float(np.mean(r2s)),
            "between": M.r2(np.array(pm), np.array(tm)), "spear": float(np.mean(sps)),
            "within_sites": dict(zip(sites, r2s))}


def main():
    grids = load_grids()
    if not grids:
        sys.exit("No grids with physical features — run add_physical_features.py first.")
    print(f"target={TGT}, sites={list(grids)}\n")
    print(f"{'features':<10}{'GLOBAL R²':>11}{'WITHIN R²':>11}{'BETWEEN R²':>12}{'Spearman':>10}")
    print("-" * 54)
    best = None
    for kind in ["base", "+Lband", "+terr", "+both", "phys"]:
        r = loso(grids, kind)
        print(f"{kind:<10}{r['global']:>+11.3f}{r['within']:>+11.3f}{r['between']:>+12.3f}{r['spear']:>+10.3f}",
              flush=True)
        if kind == "+both":
            best = r
    if best:
        print("\nper-site WITHIN R² (+both):")
        print("  " + "  ".join(f"{s}:{v:+.2f}" for s, v in best["within_sites"].items()))
    print("\nIf WITHIN R² rises toward 0+ with +Lband, L-band SAR is supplying real under-canopy info.")


if __name__ == "__main__":
    main()
