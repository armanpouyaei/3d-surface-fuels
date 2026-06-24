"""Train the PORTABLE Stage-1 model on US sites, for global 'generate anywhere'.

Builds 10 m training samples (near-ground structure target + AlphaEarth + Sentinel-1)
from our US 3DEP sites, pools them, fits a quantile GBM + conformal + OOD stats, and
saves data/processed/stage1_portable.joblib (loaded by surface_fuels.portable at
inference). Predictors are GLOBAL/FREE (AEF + S1) so the model applies anywhere; the
target is US-only (that's the honest validation boundary).

Reports leave-one-site-out R² as the generality number.

Usage:  python scripts/train_portable.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, GeoRef, lidar, metrics as M, embeddings as emb, sar, portable  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = 10.0  # AlphaEarth-native training resolution

SITES = [
    {"name": "osbs", "laz": "data/raw/osbs_3dep_2018.laz", "src": 6438, "tgt": 32617, "year": 2018, "z": "ft_us"},
    {"name": "soap", "laz": "data/raw/soap_3dep_2022.laz", "src": 6340, "tgt": 32611, "year": 2022, "z": "m"},
]


def site_samples(s):
    laz = os.path.join(ROOT, s["laz"])
    if not os.path.exists(laz):
        print(f"  [{s['name']}] missing {laz} — skipping"); return None
    print(f"  [{s['name']}] loading 3DEP + building 10 m structure + AEF({s['year']})+S1…", flush=True)
    pc = lidar.load_points(laz, src_epsg=s["src"], target_epsg=s["tgt"], z_unit=s["z"])
    x0, y0 = float(pc.x.min()), float(pc.y.min())
    nx = int((pc.x.max() - x0) // RES); ny = int((pc.y.max() - y0) // RES)
    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, max(nx, ny) * RES, res=5.0)
    gi = np.clip(((pc.x - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((pc.y - y0) / dres).astype(int), 0, ng - 1)
    h = pc.z - dtm[gj, gi]
    ix = np.clip(((pc.x - x0) / RES).astype(int), 0, nx - 1)
    iy = np.clip(((pc.y - y0) / RES).astype(int), 0, ny - 1)
    surf = (h >= 0.15) & (h < 4.0)
    near = np.zeros((ny, nx), np.float32); np.add.at(near, (iy[surf], ix[surf]), 1.0)
    tot = np.zeros((ny, nx), np.float32); np.add.at(tot, (iy, ix), 1.0)
    frac = np.divide(near, tot, out=np.zeros_like(near), where=tot > 0)
    target = (frac * (0.6 / max(frac[frac > 0].mean(), 1e-6))).astype(np.float32)  # -> ~0.6 kg/m² mean

    grid = FuelVoxelGrid(np.zeros((1, ny, nx), np.float32), dz=RES, dy=RES, dx=RES,
                         georef=GeoRef(f"EPSG:{s['tgt']}", x0, y0, RES))
    aef = emb.aef_for_grid(grid, year=s["year"])
    s1 = sar.sar_features_for_grid(grid)
    if aef is None:
        print(f"  [{s['name']}] AEF fetch failed — skipping"); return None
    cols = [np.nan_to_num(aef[i]).ravel() for i in range(64)]
    if s1 is not None:
        cols += [np.nan_to_num(s1["vh_vv"]).ravel(), np.nan_to_num(s1["rvi"]).ravel()]
    else:
        cols += [np.zeros(ny * nx, np.float32)] * 2
    X = np.stack(cols, 1).astype(np.float32)
    y = target.ravel()
    keep = (tot.ravel() > 0)
    print(f"  [{s['name']}] {keep.sum()} samples @10 m", flush=True)
    return X[keep], y[keep]


def main():
    data = {}
    for s in SITES:
        r = site_samples(s)
        if r is not None:
            data[s["name"]] = r
    if not data:
        sys.exit("No training data built.")

    # leave-one-site-out generality
    print("\nLeave-one-site-out generality (train others → predict held-out site):")
    names = list(data)
    for held in names:
        if len(names) < 2:
            break
        Xtr = np.vstack([data[n][0] for n in names if n != held])
        ytr = np.concatenate([data[n][1] for n in names if n != held])
        Xte, yte = data[held]
        mdl = g_fit(Xtr, ytr)
        pred = np.clip(mdl.predict(Xte), 0, None)
        print(f"  train {[n for n in names if n != held]} → {held}: R² {M.r2(pred, yte):.3f} "
              f"(n_test {len(yte)})")

    X = np.vstack([data[n][0] for n in names])
    y = np.concatenate([data[n][1] for n in names])
    path = portable.train(X, y, sites=names)
    print(f"\nSaved portable model ({len(y)} pooled samples, sites {names}) → {path}")
    print("Inference: surface_fuels.portable.predict_aoi(lat, lon) — AEF+S1 only, no LiDAR.")


def g_fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="squared_error", max_depth=6, learning_rate=0.08,
                                         max_iter=150, l2_regularization=1.0, random_state=0).fit(X, y)


if __name__ == "__main__":
    main()
