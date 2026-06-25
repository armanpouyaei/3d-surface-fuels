"""(c) Does the v3 occupancy product beat FastFuels' uniform-per-class surface layer WITHIN a
seen ecosystem? FastFuels assigns ONE value per 30 m fuel-model class -> zero heterogeneity
below the class. We test, on cached seen forest sites, with SPATIALLY-BLOCKED CV (no leakage):

  ours      blocked-CV GBM prediction of occ_thin from v3 features (AEF+S1+L-band), at 10 m
  uniform   the FastFuels analog = constant within each 30 m block (block-mean) -> within-block
            heterogeneity is 0 BY CONSTRUCTION (FastFuels is even flatter: uniform per CLASS,
            which spans many blocks — so this is a conservative, FastFuels-favouring baseline)

Headline metric (matches §1): WITHIN-BLOCK R² (block mean removed) — the sub-class heterogeneity
FastFuels structurally cannot represent. Plus CV(ours) vs CV(truth) vs CV(uniform).

Run:  python scripts/validate_v3_vs_fastfuels.py
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
SEEN = ["osbs", "soap", "wref", "harv", "fr"]   # forested seen ecosystems (real understory to resolve)
BLK = 3            # 30 m fuel-class block = 3x3 of 10 m cells
NFOLD = 3          # split the site into NFOLD x NFOLD super-blocks for spatial CV


def load(site):
    f = os.path.join(ROOT, "data", "interim", f"portable_grid_{site}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def feats(g):
    return [np.nan_to_num(g["aef"][k]) for k in range(64)] + \
           [np.nan_to_num(g["s1"][0]), np.nan_to_num(g["s1"][1]),
            np.nan_to_num(g["pal_hh"]), np.nan_to_num(g["pal_hv"]), np.nan_to_num(g["pal_ratio"])]


def gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6,
                                         learning_rate=0.08, max_iter=300, l2_regularization=1.0)


def block_mean_map(z, valid):
    """Per-30 m-block mean broadcast back to 10 m (the uniform / FastFuels analog)."""
    ny, nx = z.shape
    out = np.zeros_like(z)
    for by in range(0, ny, BLK):
        for bx in range(0, nx, BLK):
            sl = (slice(by, by + BLK), slice(bx, bx + BLK))
            m = valid[sl]
            if m.any():
                out[sl] = z[sl][m].mean()
    return out


def blocked_cv_predict(g):
    """Predict occ_thin at every valid cell via spatial NFOLDxNFOLD super-block CV."""
    ny, nx = g["target"].shape
    valid = g["valid"]
    F = np.stack([a.ravel() for a in feats(g)], 1).astype(np.float32)
    y = g["occ_thin"].ravel().astype(np.float32)
    # super-block fold id per cell
    yi, xi = np.mgrid[0:ny, 0:nx]
    fold = ((yi * NFOLD // ny) * NFOLD + (xi * NFOLD // nx)).ravel()
    pred = np.full(ny * nx, np.nan, np.float32)
    v = valid.ravel()
    for k in np.unique(fold):
        tr = v & (fold != k); te = v & (fold == k)
        if tr.sum() < 50 or te.sum() == 0:
            continue
        pred[te] = np.clip(gbm().fit(F[tr], y[tr]).predict(F[te]), 0, None)
    return pred.reshape(ny, nx)


def within_block_r2(pred, truth, valid):
    """R² after removing each 30 m block's mean from BOTH pred and truth (sub-class signal)."""
    pb = block_mean_map(pred, valid); tb = block_mean_map(truth, valid)
    m = valid & np.isfinite(pred)
    return M.r2((pred - pb)[m], (truth - tb)[m])


def cv(z, valid):
    v = z[valid & np.isfinite(z)]
    return float(v.std() / (v.mean() + 1e-9))


def main():
    print(f"v3 (occupancy) vs FastFuels-uniform WITHIN seen ecosystems — blocked CV, 30 m class blocks\n")
    print(f"{'site':<6}{'overall R²':>11}{'WITHIN-blk R²':>14}{'uniform within':>15}"
          f"{'CV ours':>9}{'CV truth':>10}{'CV unif':>9}")
    print("-" * 78)
    wb_ours, wb_unif = [], []
    for s in SEEN:
        g = load(s)
        if g is None or "occ_thin" not in g or "pal_hv" not in g:
            print(f"{s:<6}  (missing occ_thin/L-band)"); continue
        truth = g["occ_thin"]; valid = g["valid"]
        pred = blocked_cv_predict(g)
        m = valid & np.isfinite(pred)
        overall = M.r2(pred[m], truth[m])
        wb = within_block_r2(pred, truth, valid)
        unif = block_mean_map(truth, valid)
        wb_u = within_block_r2(unif, truth, valid)            # = 0 by construction (sanity)
        print(f"{s:<6}{overall:>+11.3f}{wb:>+14.3f}{wb_u:>+15.3f}"
              f"{cv(pred, valid):>9.2f}{cv(truth, valid):>10.2f}{cv(unif, valid):>9.2f}", flush=True)
        wb_ours.append(wb); wb_unif.append(wb_u)
    if wb_ours:
        print(f"\nMEAN within-block R²:  ours {np.mean(wb_ours):+.3f}   FastFuels-uniform {np.mean(wb_unif):+.3f}")
        print("Ours > 0 = we resolve sub-class heterogeneity FastFuels' uniform layer cannot (its")
        print("within-block R² is 0 by construction, and it is even flatter — uniform per CLASS, not per block).")


if __name__ == "__main__":
    main()
