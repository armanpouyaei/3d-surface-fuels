"""Ingest + visualize a 3D Surface Fuels deliverable NetCDF (challenge Option C / Option-1 arrays).

Standalone tool the judges can run in ONE command: it reads a submission NetCDF (3D bulk density +
fuel-property arrays), prints a property summary and a validation "smell test", and renders
property maps + a 3D voxel view. Reads the sibling AOI-boundary GeoJSON if present.

Usage:
    python scripts/read_deliverable.py PATH.nc [--boundary PATH.geojson] [--out DIR]

Only needs numpy / xarray / matplotlib (see requirements.txt). Works on any file written by
surface_fuels.FuelVoxelGrid.to_netcdf (measured OSBS product or a generate-anywhere export).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

SENTINEL = 1.23456


def _stats(a):
    a = np.asarray(a, float)
    m = np.isfinite(a) & (np.abs(a - SENTINEL) > 1e-6)   # ignore the no-data sentinel
    if not m.any():
        return "  (all no-data)"
    v = a[m]
    return f"mean {v.mean():.3f}  min {v.min():.3f}  max {v.max():.3f}  nonzero {100*np.mean(v>0):.0f}%"


def main():
    ap = argparse.ArgumentParser(description="Read + visualize a 3D surface-fuels deliverable NetCDF.")
    ap.add_argument("netcdf", help="path to the deliverable .nc")
    ap.add_argument("--boundary", help="AOI boundary GeoJSON (default: sibling *_boundary.geojson)")
    ap.add_argument("--out", help="figure output dir (default: alongside the .nc)")
    args = ap.parse_args()

    import xarray as xr
    from surface_fuels import FuelVoxelGrid
    ds = xr.open_dataset(args.netcdf)
    g = FuelVoxelGrid.from_netcdf(args.netcdf)
    outdir = args.out or os.path.dirname(os.path.abspath(args.netcdf))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.netcdf))[0]

    # ---- provenance + geometry ----
    print(f"\n=== {os.path.basename(args.netcdf)} ===")
    print(f"grid: {g.nz}×{g.ny}×{g.nx} (z,y,x)  voxel {g.dz:g}×{g.dy:g}×{g.dx:g} m  CRS {g.georef.crs}")
    print(f"origin (x0,y0) = ({g.georef.x0:.1f}, {g.georef.y0:.1f})  no-data sentinel {SENTINEL}")
    for k in ("title", "scenario", "source", "submission_pathway", "note"):
        if k in ds.attrs:
            print(f"  {k}: {ds.attrs[k]}")

    # ---- property table (every variable in the file) ----
    print("\nfuel properties:")
    for v in ds.data_vars:
        u = ds[v].attrs.get("units", "")
        ln = ds[v].attrs.get("long_name", "")
        print(f"  {v:<15} [{u:^8}] {ln}\n      {_stats(ds[v].values)}")

    # ---- validation smell test ----
    load = g.fuel_load()
    print("\nsmell test:")
    print(f"  mean load {load.mean():.3f} kg/m²  · load CV {load.std()/(load.mean()+1e-9):.2f} "
          f"(heterogeneity; 0 = uniform/FastFuels-style)")
    print(f"  fuelbed depth {g.fuelbed_depth().mean():.2f} m  · column occupancy {g.occupancy().mean()*100:.0f}%")
    p3 = {k: v for k, v in ds.attrs.items() if k.startswith("p3_")}
    if p3:
        print("  P3 landscape metrics: " + " · ".join(f"{k[3:]} {v}" for k, v in p3.items()))

    # ---- boundary ----
    bpath = args.boundary or (os.path.splitext(args.netcdf)[0] + "_boundary.geojson")
    if not os.path.exists(bpath):
        cand = glob.glob(os.path.join(os.path.dirname(args.netcdf) or ".", "*boundary*.geojson"))
        bpath = cand[0] if cand else None
    if bpath and os.path.exists(bpath):
        bj = json.load(open(bpath))
        ring = bj["features"][0]["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
        print(f"\nAOI boundary ({os.path.basename(bpath)}): lon {min(lons):.4f}..{max(lons):.4f}  "
              f"lat {min(lats):.4f}..{max(lats):.4f}  native {bj['features'][0]['properties'].get('native_crs')}")

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    flip = np.flipud
    maps = [("fuel_load", "kg/m²", "YlOrBr"), ("percent_cover", "%", "Greens"),
            ("bulk_density_maxz", "kg/m³", "magma"), ("fuelbed_depth", "m", "viridis")]
    arrs = {"fuel_load": load, "percent_cover": 100 * g.occupancy().mean(0),
            "bulk_density_maxz": g.bulk_density.max(0), "fuelbed_depth": g.fuelbed_depth()}
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a, (name, u, cm) in zip(ax, maps):
        arr = arrs[name]
        im = a.imshow(flip(arr), cmap=cm, aspect="equal", vmin=0, vmax=float(np.percentile(arr, 99) + 1e-6))
        a.set_title(f"{name} ({u})", fontsize=10); a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle(f"{base} — fuel property maps", fontsize=12)
    mp = os.path.join(outdir, f"{base}_maps.png"); fig.tight_layout(); fig.savefig(mp, dpi=110); plt.close(fig)

    # 3D voxel view (coarsen so it renders fast)
    bd = g.bulk_density
    step = max(1, g.nx // 60)
    bdc = bd[:, ::step, ::step]
    zz, yy, xx = np.where(bdc > 0)
    fig = plt.figure(figsize=(7, 6)); a3 = fig.add_subplot(111, projection="3d")
    if len(zz):
        p = a3.scatter(xx, yy, zz, c=bdc[zz, yy, xx], cmap="magma", s=6, marker="s")
        plt.colorbar(p, ax=a3, fraction=0.03, pad=0.08, label="bulk density (kg/m³)")
    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z (m)"); a3.set_title(f"{base} — 3D fuel voxels")
    tp = os.path.join(outdir, f"{base}_3d.png"); fig.tight_layout(); fig.savefig(tp, dpi=110); plt.close(fig)

    print(f"\nwrote {os.path.relpath(mp)}\nwrote {os.path.relpath(tp)}")
    print("done.")


if __name__ == "__main__":
    main()
