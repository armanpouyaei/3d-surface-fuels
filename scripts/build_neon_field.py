"""Validate spaceborne predictors against REAL NEON field fuel load (Stage-1, #1).

Uses NEON OSBS herbaceous clip-harvest loads (data/interim/neon_osbs_herb.csv,
79 clips / 20 plots, 2018-19) as the field target. Samples AlphaEarth (2018) +
Sentinel-1 at each clip location and tests, with leave-one-PLOT-out CV, whether
free spaceborne data predicts field-measured herbaceous surface fuel.

Honest caveats: small n (~20 spatial plots), herb component only, and a scale
mismatch (0.2 m² clip vs 10 m pixel) that caps achievable skill. Figures saved
to figures/. Usage: python scripts/build_neon_field.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from surface_fuels import FuelVoxelGrid, GeoRef, embeddings as emb, sar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
EPSG = 32617  # UTM 17N (OSBS)


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "interim", "neon_osbs_herb.csv"))
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    d["x"], d["y"] = tr.transform(d.decimalLongitude.values, d.decimalLatitude.values)

    res, buf = 10.0, 200.0
    x0, y0 = d.x.min() - buf, d.y.min() - buf
    nx = int((d.x.max() + buf - x0) // res) + 1
    ny = int((d.y.max() + buf - y0) // res) + 1
    grid = FuelVoxelGrid(np.zeros((1, ny, nx), np.float32), dz=res, dy=res, dx=res,
                         georef=GeoRef(f"EPSG:{EPSG}", x0, y0, res))
    print(f"{len(d)} clips / {d.plotID.nunique()} plots; AOI {nx}x{ny} @ {res:.0f} m")

    print("Fetching AlphaEarth (2018) + Sentinel-1 over the clip AOI...")
    aef = emb.aef_for_grid(grid, year=2018)
    s1 = sar.sar_features_for_grid(grid)
    col = np.clip(((d.x - x0) / res).astype(int), 0, nx - 1).values
    row = np.clip(((d.y - y0) / res).astype(int), 0, ny - 1).values  # row 0 = south (matches grids)

    AEF = np.stack([np.nan_to_num(aef[:, r, c]) for r, c in zip(row, col)]) if aef is not None else None
    S1 = np.column_stack([s1["vh_vv"][row, col], s1["rvi"][row, col]]) if s1 is not None else None
    y = d.load_kg_m2.values
    groups = d.plotID.values

    # leave-one-plot-out CV with a regularized linear model
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from scipy.stats import pearsonr, spearmanr
    from surface_fuels import metrics as M

    def lopo(X):
        pred = np.zeros_like(y)
        for tr_i, te_i in LeaveOneGroupOut().split(X, y, groups):
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr_i], y[tr_i])
            pred[te_i] = m.predict(X[te_i])
        return np.clip(pred, 0, None)

    sets = {}
    if AEF is not None and S1 is not None:
        sets["full (AEF+S1)"] = np.column_stack([AEF, S1])
    if AEF is not None:
        sets["AEF"] = AEF
    if S1 is not None:
        sets["S1"] = S1
    reps = {k: lopo(X) for k, X in sets.items()}
    print("\nLeave-one-plot-out (predict field herb load kg/m²):")
    for k, p in reps.items():
        print(f"  {k:<14} R2 {M.r2(p, y):>6.3f}  Spearman {spearmanr(p, y).correlation:>5.2f}  RMSE {M.rmse(p, y):.3f}")

    # univariate correlations (which AEF bands / S1 carry signal)
    corrs = []
    if AEF is not None:
        for i in range(AEF.shape[1]):
            corrs.append((f"AEF{i:02d}", pearsonr(AEF[:, i], y)[0]))
    if S1 is not None:
        corrs += [("S1 VH/VV", pearsonr(S1[:, 0], y)[0]), ("S1 RVI", pearsonr(S1[:, 1], y)[0])]
    corrs.sort(key=lambda t: -abs(t[1]))

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    top = corrs[:12]
    ax[0].barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color="#2c7fb8")
    ax[0].axvline(0, color="k", lw=.6); ax[0].set_xlabel("Pearson r with field load")
    ax[0].set_title("Top predictors vs field herb load")
    best = corrs[0]; bi = [t[0] for t in corrs].index(best[0])
    bx = AEF[:, int(best[0][3:])] if best[0].startswith("AEF") else S1[:, 0 if "VH" in best[0] else 1]
    ax[1].scatter(bx, y, s=40, alpha=.6, color="#d95f0e", edgecolor="k", lw=.3)
    ax[1].set_xlabel(best[0]); ax[1].set_ylabel("field herb load (kg/m²)")
    ax[1].set_title(f"best single predictor: {best[0]} (r={best[1]:.2f})")
    bestset = max(reps, key=lambda k: M.r2(reps[k], y))
    p = reps[bestset]
    ax[2].scatter(y, p, s=40, alpha=.6, color="#31a354", edgecolor="k", lw=.3)
    lim = [0, max(y.max(), p.max())]; ax[2].plot(lim, lim, "r--", lw=1)
    ax[2].set_xlabel("field herb load (kg/m²)"); ax[2].set_ylabel("predicted")
    ax[2].set_title(f"LOPO {bestset}: R²={M.r2(p, y):.2f}, Spearman={spearmanr(p, y).correlation:.2f}")
    plt.tight_layout()
    fp = os.path.join(ROOT, "figures", "neon_field_validation.png")
    plt.savefig(fp, dpi=110); print(f"\nsaved {fp}")
    print("Caveat: herb component only, n≈20 plots, 0.2 m² clip vs 10 m pixel scale mismatch caps skill.")


if __name__ == "__main__":
    main()
