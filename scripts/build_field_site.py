"""Build a per-site AlphaEarth + Sentinel-1 feature table at NEON herb-clip
locations (field-calibrated Stage-1, multi-ecosystem). One site -> one CSV in
data/interim/field_<site>.csv. Run across sites, then analyze with
scripts/analyze_field_multisite.py.

Usage: python scripts/build_field_site.py --site KONZ [--year 2021]
"""

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from surface_fuels import FuelVoxelGrid, GeoRef, embeddings as emb, sar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DP = "DP1.10023.001"


def utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def herb_table(site, month, tab):
    j = json.load(io.BytesIO(requests.get(
        f"https://data.neonscience.org/api/v0/data/{DP}/{site}/{month}", timeout=60).content))["data"]["files"]
    url = [f["url"] for f in j if tab in f["name"] and f["name"].endswith(".csv")][0]
    return pd.read_csv(io.BytesIO(requests.get(url, timeout=60).content))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--year", type=int, default=None, help="default: latest available <=2023")
    args = ap.parse_args()

    prod = json.load(io.BytesIO(requests.get(
        f"https://data.neonscience.org/api/v0/products/{DP}", timeout=60).content))["data"]
    months = next(s["availableMonths"] for s in prod["siteCodes"] if s["siteCode"] == args.site)
    cand = [m for m in months if int(m[:4]) <= (args.year or 2023)]
    if args.year:
        cand = [m for m in months if m.startswith(str(args.year))]
    if not cand:
        sys.exit(f"No herb-clip month for {args.site} (<= {args.year or 2023}); available: {months}")
    month = sorted(cand)[-1]
    year = int(month[:4])
    print(f"[{args.site}] herb clip {month}")

    mass = herb_table(args.site, month, "hbp_massdata")
    per = herb_table(args.site, month, "hbp_perbout")
    tot = mass.groupby("sampleID")["dryMass"].sum().rename("dryMass_g").reset_index()
    d = per.merge(tot, on="sampleID", how="inner")
    d = d.dropna(subset=["decimalLatitude", "decimalLongitude", "clipArea"])
    d["load_kg_m2"] = d.dryMass_g / d.clipArea / 1000.0
    epsg = utm_epsg(d.decimalLongitude.mean(), d.decimalLatitude.mean())
    from pyproj import Transformer
    d["x"], d["y"] = Transformer.from_crs(4326, epsg, always_xy=True).transform(
        d.decimalLongitude.values, d.decimalLatitude.values)
    print(f"  {len(d)} clips / {d.plotID.nunique()} plots; UTM EPSG:{epsg}; "
          f"load mean {d.load_kg_m2.mean():.3f} kg/m²; AEF year {year}")

    res, buf = 10.0, 200.0
    x0, y0 = d.x.min() - buf, d.y.min() - buf
    nx = int((d.x.max() + buf - x0) // res) + 1
    ny = int((d.y.max() + buf - y0) // res) + 1
    grid = FuelVoxelGrid(np.zeros((1, ny, nx), np.float32), dz=res, dy=res, dx=res,
                         georef=GeoRef(f"EPSG:{epsg}", x0, y0, res))
    print(f"  AOI {nx}x{ny} @ {res:.0f} m; fetching AlphaEarth({year}) + Sentinel-1...")
    aef = emb.aef_for_grid(grid, year=year)
    s1 = sar.sar_features_for_grid(grid)
    if aef is None:
        sys.exit("AEF fetch failed")
    col = np.clip(((d.x - x0) / res).astype(int), 0, nx - 1).values
    row = np.clip(((d.y - y0) / res).astype(int), 0, ny - 1).values

    out = pd.DataFrame({"site": args.site, "year": year, "plotID": d.plotID.values,
                        "x": d.x.values, "y": d.y.values, "load_kg_m2": d.load_kg_m2.values})
    for i in range(64):
        out[f"aef{i:02d}"] = [np.nan_to_num(aef[i, r, c]) for r, c in zip(row, col)]
    if s1 is not None:
        out["s1_vhvv"] = s1["vh_vv"][row, col]
        out["s1_rvi"] = s1["rvi"][row, col]
    os.makedirs(os.path.join(ROOT, "data", "interim"), exist_ok=True)
    fp = os.path.join(ROOT, "data", "interim", f"field_{args.site}.csv")
    out.to_csv(fp, index=False)
    print(f"  saved {fp} ({len(out)} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
