"""CREDIBILITY GATE for the occ_vert breakthrough: is the between-biome signal real
vegetation, or a LiDAR point-density artifact? Re-derive vertical occupancy after THINNING
every site to a common density (so denser surveys can't fake higher occupancy), append as
occ_thin / cover_thin to the cached grids. If occ_thin keeps high BETWEEN/GLOBAL R² in
eval_targets, the signal is vegetation, not the sensor.

Common density floor = 2.0 pts/m² (≈ the sparsest site, cper). Each site's points are
randomly kept with prob min(1, 2.0/site_density). Memory-safe: one LAZ at a time.

Run:  python scripts/rederive_thinned.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import lidar  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from train_portable import SITES, RES, ROOT  # noqa: E402

LOW, HIGH = 0.15, 4.0
NBIN = int((HIGH - LOW) / 0.5)
TARGET_DENS = 2.0          # pts/m² common floor
SEED = 0


def derive(s):
    laz = os.path.join(ROOT, s["laz"])
    gp = os.path.join(ROOT, "data", "interim", f"portable_grid_{s['name']}.npz")
    if not (os.path.exists(gp) and os.path.exists(laz)):
        print(f"  [{s['name']}] missing — skip"); return
    d = dict(np.load(gp))
    if "occ_thin" in d:
        print(f"  [{s['name']}] already thinned — skip"); return
    ny, nx = d["target"].shape

    pc = lidar.load_points(laz, src_epsg=s["src"], target_epsg=s["tgt"], z_unit=s["z"])
    x0, y0 = float(pc.x.min()), float(pc.y.min())
    area = (pc.x.max() - x0) * (pc.y.max() - y0)
    dens = len(pc.x) / max(area, 1.0)
    keep_p = min(1.0, TARGET_DENS / max(dens, 1e-6))
    rng = np.random.RandomState(SEED)
    m = rng.random(len(pc.x)) < keep_p
    print(f"  [{s['name']}] dens {dens:.1f}→thin keep {keep_p:.3f} ({m.sum()} pts)", flush=True)

    px, py, pz = pc.x[m], pc.y[m], pc.z[m]
    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, max(nx, ny) * RES, res=5.0)
    gi = np.clip(((px - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((py - y0) / dres).astype(int), 0, ng - 1)
    h = pz - dtm[gj, gi]
    ix = np.clip(((px - x0) / RES).astype(int), 0, nx - 1)
    iy = np.clip(((py - y0) / RES).astype(int), 0, ny - 1)

    near_m = (h >= LOW) & (h < HIGH)
    floor_m = (h >= 0.0) & (h < LOW)
    n_near = np.zeros((ny, nx), np.float32); np.add.at(n_near, (iy[near_m], ix[near_m]), 1.0)
    n_floor = np.zeros((ny, nx), np.float32); np.add.at(n_floor, (iy[floor_m], ix[floor_m]), 1.0)
    occ = np.zeros((ny, nx), np.float32)
    for b in range(NBIN):
        lo, hi = LOW + b * 0.5, LOW + (b + 1) * 0.5
        mb = (h >= lo) & (h < hi)
        bh = np.zeros((ny, nx), bool); bh[iy[mb], ix[mb]] = True
        occ += bh
    occ /= NBIN
    denom = n_near + n_floor
    cover = np.where(denom > 0, n_near / denom, 0.0).astype(np.float32)

    d.update(occ_thin=occ.astype(np.float32), cover_thin=cover)
    np.savez_compressed(gp, **d)
    v = d["valid"]
    print(f"    occ_thin μ{occ[v].mean():.3f}  cover_thin μ{cover[v].mean():.3f}", flush=True)
    del pc, h, ix, iy, gi, gj, px, py, pz


def main():
    for s in SITES:
        try:
            derive(s)
        except Exception as e:
            print(f"  [{s['name']}] FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
