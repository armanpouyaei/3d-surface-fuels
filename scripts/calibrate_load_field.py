"""Field-calibrate occupancy -> surface LOAD (kg/m²) against RxCADRE destructive clip-plot loads
(the gold-standard, total-surface, co-located truth at Eglin). For each 2012 Eglin burn block:
  field load = mean pre-fire Pre_loading_metric_rebinned_* (Mg/ha -> kg/m²) over its clip plots
  our occ   = v3 model mean predicted vertical occupancy at the block centroid
Fit occ -> load (through the origin) for TOTAL and for the ABOVE-LITTER components (herb+shrub+
1h/10h/100h, which match our 0.15-4 m stratum). The fitted slope replaces the literature
OCC_TO_LOAD anchor. RxCADRE loads RDS-2014-0028 + coords RDS-2014-0030 (UTM 16N).

Run:  python scripts/calibrate_load_field.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyogrio.raw import read  # noqa: E402
from pyproj import Transformer  # noqa: E402

from surface_fuels import portable, metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
GDB = os.path.join(ROOT, "data/raw/rxcadre/coords/Data/RxCADRE_clipplot_locs.gdb")
LOADS = os.path.join(ROOT, "data/raw/rxcadre/loads/Data")
MAIN_BLOCKS = ["L1G", "L2F", "L2G", "S3", "S4", "S5", "S7", "S8", "S9"]  # Eglin 2012 clip-plot blocks
ABOVE = ["Pre_loading_metric_rebinned_herb", "Pre_loading_metric_rebinned_shrub",
         "Pre_loading_metric_rebinned_woody_1hr_10hr_100hr_cone"]   # ~0.15-4 m stratum


def block_coords():
    out = read(GDB, layer="RxCADRE_2012_clipplot_locs")
    names, fd = out[0]["fields"], out[-1]
    d = pd.DataFrame({n: f for n, f in zip(names, fd)})
    d = d[d["Description"].astype(str).str.upper() == "PRE"]
    tr = Transformer.from_crs(26916, 4326, always_xy=True)
    cen = {}
    for blk, g in d.groupby(d["Block"].astype(str)):
        e, n = g["Easting"].astype(float).mean(), g["Northing"].astype(float).mean()
        lon, lat = tr.transform(e, n)
        cen[blk] = (lat, lon, len(g))
    return cen


def block_load(blk):
    f = os.path.join(LOADS, f"2012_{blk}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    df = df[df["Category"].astype(str).str.startswith("Plot_")]
    tot = pd.to_numeric(df["Pre_loading_metric_rebinned_TOTAL"], errors="coerce")
    above = sum(pd.to_numeric(df[c], errors="coerce") for c in ABOVE)
    return float(np.nanmean(tot)) / 10.0, float(np.nanmean(above)) / 10.0   # Mg/ha -> kg/m²


def fit_origin(x, y):
    x, y = np.asarray(x), np.asarray(y)
    k = float((x * y).sum() / (x * x).sum())          # least-squares slope through origin
    return k, M.r2(k * x, y)


def main():
    cen = block_coords()
    rows = []
    print(f"{'block':<6}{'centroid lat,lon':<22}{'occ':>6}{'field TOTAL':>12}{'above-litter':>13}")
    print("-" * 60)
    for blk in MAIN_BLOCKS:
        if blk not in cen:
            continue
        lat, lon, npl = cen[blk]
        ld = block_load(blk)
        if ld is None:
            continue
        total, above = ld
        r = portable.predict_aoi(lat, lon, size=300.0, year=2017, res=10, use_cache=True, version="v3")
        occ = float(r["pred10"].mean())
        rows.append((blk, occ, total, above))
        print(f"{blk:<6}{f'{lat:.4f}, {lon:.4f}':<22}{occ:>6.2f}{total:>12.2f}{above:>13.2f}", flush=True)
    if len(rows) < 3:
        print("too few blocks"); return
    occ = np.array([r[1] for r in rows]); tot = np.array([r[2] for r in rows]); ab = np.array([r[3] for r in rows])
    kt, r2t = fit_origin(occ, tot)
    ka, r2a = fit_origin(occ, ab)
    print(f"\nField calibration (occ -> load through origin, n={len(rows)} Eglin blocks):")
    print(f"  TOTAL surface load:  K = {kt:.2f} kg/m² per unit occ   (R² {r2t:+.2f})   "
          f"[mean field {tot.mean():.2f}, mean occ {occ.mean():.2f}]")
    print(f"  ABOVE-LITTER (0.15-4 m): K = {ka:.2f}   (R² {r2a:+.2f})   [mean field {ab.mean():.2f}]")
    print(f"\n  current literature anchor OCC_TO_LOAD_KG_M2 = {portable.OCC_TO_LOAD_KG_M2:.2f}")
    print(f"  -> field-calibrated K (TOTAL) = {kt:.2f}  (×{kt/portable.OCC_TO_LOAD_KG_M2:.2f} vs current)")
    import json
    json.dump({"K_total": kt, "R2_total": r2t, "K_above": ka, "R2_above": r2a,
               "blocks": [r[0] for r in rows], "mean_field_total": float(tot.mean()),
               "mean_occ": float(occ.mean())},
              open(os.path.join(ROOT, "data/interim/rxcadre_calibration.json"), "w"), indent=2)
    print("  saved data/interim/rxcadre_calibration.json")


if __name__ == "__main__":
    main()
