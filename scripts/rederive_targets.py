"""Re-derive several candidate 10 m understory-structure targets from the LAZ files and
append them to each cached portable_grid_*.npz (keeps AEF/S1 registered — no re-fetch).
Memory-safe: one LAZ at a time, point cloud freed between sites.

Candidate targets per 10 m cell (all on the SAME grid as the cached AEF/S1):
  frac_raw  near-ground return fraction n_near/n_total           (current proxy, PRE-normalization)
  cover     understory cover among low returns n_near/(n_near+n_floor)  (ratio -> density-robust)
  occ_vert  vertical occupancy: frac of 0.5 m bins in [0.15,4) with >=1 return  (presence -> density-robust)
  pad_gap   Beer-Lambert understory PAD  -ln(gap),  gap=n_floor/(n_floor+n_near)  (sensor-robust)
  dens      point density (pts/m^2)  -- stored to MEASURE the sensor confound, not as a target

Run:  python scripts/rederive_targets.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import lidar  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from train_portable import SITES, RES, ROOT  # noqa: E402

LOW, HIGH = 0.15, 4.0          # understory stratum (m above ground)
FLOOR = 0.15                   # below this = ground/litter floor
NBIN = int((HIGH - LOW) / 0.5)  # 0.5 m vertical bins


def derive(s):
    laz = os.path.join(ROOT, s["laz"])
    gp = os.path.join(ROOT, "data", "interim", f"portable_grid_{s['name']}.npz")
    if not os.path.exists(gp):
        print(f"  [{s['name']}] no cached grid — skip"); return
    if not os.path.exists(laz):
        print(f"  [{s['name']}] LAZ missing — skip"); return
    d = dict(np.load(gp))
    if "frac_raw" in d:
        print(f"  [{s['name']}] already has targets — skip"); return
    ny, nx = d["target"].shape
    print(f"  [{s['name']}] reading LAZ → {ny}×{nx} grid …", flush=True)

    pc = lidar.load_points(laz, src_epsg=s["src"], target_epsg=s["tgt"], z_unit=s["z"])
    x0, y0 = float(pc.x.min()), float(pc.y.min())
    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, max(nx, ny) * RES, res=5.0)
    gi = np.clip(((pc.x - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((pc.y - y0) / dres).astype(int), 0, ng - 1)
    h = pc.z - dtm[gj, gi]
    ix = np.clip(((pc.x - x0) / RES).astype(int), 0, nx - 1)
    iy = np.clip(((pc.y - y0) / RES).astype(int), 0, ny - 1)

    near_m = (h >= LOW) & (h < HIGH)
    floor_m = (h >= 0.0) & (h < FLOOR)
    n_near = np.zeros((ny, nx), np.float32); np.add.at(n_near, (iy[near_m], ix[near_m]), 1.0)
    n_floor = np.zeros((ny, nx), np.float32); np.add.at(n_floor, (iy[floor_m], ix[floor_m]), 1.0)
    n_tot = np.zeros((ny, nx), np.float32); np.add.at(n_tot, (iy, ix), 1.0)

    # vertical occupancy: count distinct 0.5 m bins with >=1 return in [LOW,HIGH)
    occ = np.zeros((ny, nx), np.float32)
    for b in range(NBIN):
        lo, hi = LOW + b * 0.5, LOW + (b + 1) * 0.5
        m = (h >= lo) & (h < hi)
        binhit = np.zeros((ny, nx), bool)
        binhit[iy[m], ix[m]] = True
        occ += binhit
    occ /= NBIN

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_raw = np.where(n_tot > 0, n_near / n_tot, 0.0).astype(np.float32)
        denom = n_near + n_floor
        cover = np.where(denom > 0, n_near / denom, 0.0).astype(np.float32)
        gap = np.where(denom > 0, n_floor / denom, 1.0)
        pad_gap = np.clip(-np.log(np.clip(gap, 1e-3, 1.0)), 0, 6).astype(np.float32)
    dens = (n_tot / (RES * RES)).astype(np.float32)

    d.update(frac_raw=frac_raw, cover=cover, occ_vert=occ.astype(np.float32),
             pad_gap=pad_gap, dens=dens)
    np.savez_compressed(gp, **d)
    v = d["valid"]
    print(f"    frac_raw μ{frac_raw[v].mean():.3f} cover μ{cover[v].mean():.3f} "
          f"occ μ{occ[v].mean():.3f} pad μ{pad_gap[v].mean():.2f} dens μ{dens[v].mean():.1f} pts/m²",
          flush=True)
    del pc, h, ix, iy, gi, gj


def main():
    for s in SITES:
        try:
            derive(s)
        except Exception as e:
            print(f"  [{s['name']}] FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
