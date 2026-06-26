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

SITES = [  # diverse ecosystems; src/tgt EPSG + z-unit + AEF year verified per LAZ header
    {"name": "osbs", "laz": "data/raw/osbs_3dep_2018.laz", "src": 6438, "tgt": 32617, "year": 2018, "z": "ft_us"},  # longleaf savanna FL
    {"name": "soap", "laz": "data/raw/soap_3dep_2022.laz", "src": 6340, "tgt": 32611, "year": 2022, "z": "m"},      # Sierra conifer CA
    {"name": "cper", "laz": "data/raw/cper_3dep.laz", "src": 6430, "tgt": 32613, "year": 2018, "z": "ft_us"},        # shortgrass steppe CO (CO N State Plane ftUS)
    {"name": "wref", "laz": "data/raw/wref_3dep.laz", "src": 6339, "tgt": 32610, "year": 2018, "z": "m"},            # PNW tall conifer WA
    {"name": "srer", "laz": "data/raw/srer_3dep.laz", "src": 6341, "tgt": 32612, "year": 2020, "z": "m"},            # desert shrub AZ
    {"name": "harv", "laz": "data/raw/harv_3dep.laz", "src": 6347, "tgt": 32618, "year": 2024, "z": "m"},            # eastern deciduous MA
    # international + biome-gap sites (open ALS, vintage-matched to AlphaEarth) — broadens coverage to 3 continents
    {"name": "ever", "laz": "data/raw/ever.laz", "src": 6346,  "tgt": 32617, "year": 2017, "z": "m"},               # Everglades wetland/swamp FL
    {"name": "ch",   "laz": "data/raw/ch.las",  "src": 2056,  "tgt": 32632, "year": 2019, "z": "m"},                # Switzerland temperate/foothill forest (swissSURFACE3D)
    {"name": "nl",   "laz": "data/raw/nl.laz",  "src": 28992, "tgt": 32631, "year": 2022, "z": "m"},                # Netherlands heath + Scots pine (AHN4)
    {"name": "fr",   "laz": "data/raw/fr.laz",  "src": 2154,  "tgt": 32631, "year": 2024, "z": "m"},                # France pre-alpine mixed forest (IGN LiDAR HD)
    # biome-coverage expansion (tropical / boreal / arid) to shrink the OOD-flagged globe
    {"name": "pr",     "laz": "data/raw/pr.laz",     "src": 6566, "tgt": 32620, "year": 2018, "z": "m"},            # tropical/subtropical broadleaf, Puerto Rico 3DEP
    {"name": "lva",    "laz": "data/raw/lva.las",    "src": 3059, "tgt": 32635, "year": 2018, "z": "m"},            # boreal/hemiboreal conifer, Latvia LGIA
    {"name": "mojave", "laz": "data/raw/mojave.laz", "src": 6350, "tgt": 32611, "year": 2019, "z": "m"},            # arid Mojave desert scrub, 3DEP
    # equatorial rainforest (two continents) — bring the tropics in-distribution
    {"name": "amz",    "laz": "data/raw/amz.laz",    "src": 31980, "tgt": 32721, "year": 2017, "z": "m"},           # Amazon terra-firme, Reserva Ducke (ORNL DAAC) — UTM 21S to match inference (Ducke is at the 20/21 boundary)
    {"name": "borneo", "laz": "data/raw/borneo.las", "src": 32650, "tgt": 32650, "year": 2020, "z": "m"},           # Bornean dipterocarp, Danum Valley (NERC CEDA)
    # akf (Alaska Fairbanks boreal, EPSG 6335) intentionally omitted — AlphaEarth mirror coverage ends ~64°N (AEF fetch fails)
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
    # cache the 2D grids so v2 (CNN patches) + re-analysis don't need to re-fetch
    s1arr = (np.stack([np.nan_to_num(s1["vh_vv"]), np.nan_to_num(s1["rvi"])]).astype(np.float32)
             if s1 is not None else np.zeros((2, ny, nx), np.float32))
    os.makedirs(os.path.join(ROOT, "data", "interim"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "data", "interim", f"portable_grid_{s['name']}.npz"),
                        target=target, aef=np.nan_to_num(aef).astype(np.float32), s1=s1arr,
                        valid=(tot > 0), x0=x0, y0=y0, epsg=s["tgt"], res=RES, year=s["year"])
    print(f"  [{s['name']}] {keep.sum()} samples @10 m (grid cached)", flush=True)
    return X[keep], y[keep]


def main():
    data = {}
    for s in SITES:
        r = site_samples(s)
        if r is not None:
            data[s["name"]] = r
    if not data:
        sys.exit("No training data built.")

    # leave-one-site-out generality — R² (absolute) AND Spearman (does the PATTERN transfer?)
    from scipy.stats import spearmanr
    print("\nLeave-one-site-out generality (train others → predict held-out site):")
    names = list(data)
    r2s, sps = [], []
    for held in names:
        if len(names) < 2:
            break
        Xtr = np.vstack([data[n][0] for n in names if n != held])
        ytr = np.concatenate([data[n][1] for n in names if n != held])
        Xte, yte = data[held]
        mdl = g_fit(Xtr, ytr)
        pred = np.clip(mdl.predict(Xte), 0, None)
        r2 = M.r2(pred, yte); sp = spearmanr(pred, yte).correlation
        r2s.append(r2); sps.append(sp)
        print(f"  → {held:5}: R² {r2:+.3f}  Spearman {sp:+.3f}  (n_test {len(yte)})")
    if r2s:
        print(f"  MEAN: R² {np.mean(r2s):+.3f}  Spearman {np.mean(sps):+.3f}  "
              "(Spearman>0 = the spatial PATTERN transfers even if absolute scale doesn't)")

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
