"""Pull FastFuels' OWN 3D voxel product over OSBS via the live v1 API.

Demonstrates the surface-vs-canopy point concretely: FastFuels' 3D structure comes
from **voxelized trees** (TreeMap), while its **surface** layer is the uniform
LANDFIRE FBFM40 lookup. Flow (verified against the live API):
  domain → TreeMap tree inventory → tree/canopy grid (bulkDensity) → [surface grid]
  → combined zarr export → download → read with xarray/zarr.

v1 has no NetCDF export, so we read the zarr and mirror it to our [z,y,x] kg/m³
convention → data/processed/ff_osbs_3d.npz (+ a NetCDF mirror for Option C interop).

Needs FASTFUELS_API_KEY. Usage:  FASTFUELS_API_KEY=... python scripts/build_fastfuels3d_osbs.py
"""

import os
import sys
import json
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
BASE = "https://api.fastfuels.silvxlabs.com"
AOI_HALF = 128.0     # 256 m AOI (1 m³ voxels → ~256×256×canopy-height)
HRES = 1.0


def call(method, path, body=None, key=None):
    import requests
    h = {"api-key": key, "Content-Type": "application/json"}
    r = requests.request(method, BASE + path, headers=h,
                         data=json.dumps(body) if body is not None else None, timeout=90)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def poll(path, key, tries=120, every=8, ok="completed"):
    for _ in range(tries):
        _, j = call("GET", path, key=key)
        st = j.get("status")
        if st == ok:
            return j
        if st in ("failed", "error", "expired", None):
            return j
        time.sleep(every)
    return {"status": "timeout"}


def main():
    key = os.environ.get("FASTFUELS_API_KEY")
    if not key:
        sys.exit("Set FASTFUELS_API_KEY to pull FastFuels' live 3D product.")
    if not os.path.exists(os.path.join(PROC, "osbs_measured_1m.nc")):
        sys.exit("Run scripts/build_deliverable_osbs.py first (need the OSBS AOI footprint).")
    g = FuelVoxelGrid.from_netcdf(os.path.join(PROC, "osbs_measured_1m.nc"))
    epsg = int(str(g.georef.crs).split(":")[-1])
    cx = g.georef.x0 + g.nx * g.dx / 2
    cy = g.georef.y0 + g.ny * g.dy / 2
    from pyproj import Transformer
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    corn = [(cx - AOI_HALF, cy - AOI_HALF), (cx + AOI_HALF, cy - AOI_HALF),
            (cx + AOI_HALF, cy + AOI_HALF), (cx - AOI_HALF, cy + AOI_HALF), (cx - AOI_HALF, cy - AOI_HALF)]
    ll = [list(tr.transform(*c)) for c in corn]
    feat = {"type": "Feature", "name": "osbs_ff3d", "horizontalResolution": HRES, "verticalResolution": 1.0,
            "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ll]}}

    print(f"Creating domain over OSBS ({2*AOI_HALF:.0f} m AOI @ {HRES:.0f} m)…", flush=True)
    s, dom = call("POST", "/v1/domains", feat, key); did = dom.get("id")
    if not did:
        sys.exit(f"domain create failed: {s} {dom}")
    print(f"  domain {did}", flush=True)

    print("Adding TreeMap tree inventory (FIA-imputed stems)…", flush=True)
    call("POST", f"/v1/domains/{did}/inventories/tree", {"sources": ["TreeMap"],
         "TreeMap": {"version": "2022", "seed": 42}}, key)
    inv = poll(f"/v1/domains/{did}/inventories/tree", key)
    print(f"  inventory status: {inv.get('status')}", flush=True)
    if inv.get("status") != "completed":
        sys.exit(f"tree inventory failed: {json.dumps(inv)[:300]}")

    print("Building tree/canopy 3D grid (voxelize stems → bulk density)…", flush=True)
    call("POST", f"/v1/domains/{did}/grids/tree", {"attributes": ["bulkDensity"],
         "bulkDensity": {"source": "TreeInventory"}}, key)
    tg = poll(f"/v1/domains/{did}/grids/tree", key)
    print(f"  tree grid status: {tg.get('status')}", flush=True)
    tree_ok = tg.get("status") == "completed"

    print("Attempting surface grid (LANDFIRE FBFM40)…", flush=True)
    call("POST", f"/v1/domains/{did}/grids/surface", {"attributes": ["fuelLoad", "fuelDepth"],
         "fuelLoad": {"source": "LANDFIRE", "product": "FBFM40", "version": "2022"},
         "fuelDepth": {"source": "LANDFIRE", "product": "FBFM40", "version": "2022"}}, key)
    sg = poll(f"/v1/domains/{did}/grids/surface", key)
    surf_ok = sg.get("status") == "completed"
    print(f"  surface grid status: {sg.get('status')}  (LANDFIRE backend; may be down)", flush=True)
    if not tree_ok and not surf_ok:
        sys.exit("Neither tree nor surface grid completed — nothing to export.")

    print("Combined 3D export (zarr)…", flush=True)
    call("POST", f"/v1/domains/{did}/grids/exports/zarr", {}, key)
    ex = poll(f"/v1/domains/{did}/grids/exports/zarr", key)
    url = ex.get("signedUrl") or ex.get("url")
    if not url:
        sys.exit(f"export failed: {json.dumps(ex)[:300]}")
    zpath = os.path.join(PROC, "ff_osbs_export.zip")
    import requests
    with requests.get(url, stream=True, timeout=300) as r, open(zpath, "wb") as f:
        for c in r.iter_content(1 << 16):
            f.write(c)
    zdir = os.path.join(PROC, "ff_osbs_export.zarr")
    os.system(f"rm -rf '{zdir}'")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(zdir)
    print(f"  downloaded + unzipped → {zdir}", flush=True)

    # ── read the zarr; find the canopy (tree) bulk-density 3D array ──
    import xarray as xr
    # the store may nest the zarr root in a subfolder; find the dir holding .zgroup/.zattrs
    root = zdir
    for dp, dns, fns in os.walk(zdir):
        if ".zgroup" in fns or ".zmetadata" in fns or any(n == ".zarray" for n in fns):
            root = dp
            break
    ds = xr.open_zarr(root)
    print("  zarr variables:", {k: tuple(v.shape) for k, v in ds.data_vars.items()}, flush=True)

    def pick(*keys):
        for v in ds.data_vars:
            lv = v.lower()
            if all(k in lv for k in keys) and ds[v].ndim == 3:
                return v
        return None
    tree_var = pick("tree", "bulk") or pick("canopy", "bulk") or pick("tree") or pick("bulk")
    out = {}
    if tree_var:
        arr = np.nan_to_num(np.asarray(ds[tree_var].values)).astype(np.float32)  # (z,y,x) expected
        out["canopy_bulk_density"] = arr
        nz = arr.shape[0]
        print(f"  FastFuels canopy bulk density '{tree_var}' {arr.shape}: "
              f"max {arr.max():.3f} kg/m³, occupied {100*(arr>0).mean():.1f}%, "
              f"top occupied z≈{int(np.argwhere(arr.sum((1,2))>0).max()) if (arr.sum((1,2))>0).any() else 0} m", flush=True)
    surf_var = pick("surface", "load") or pick("surface", "bulk")
    if surf_var:
        sa = np.nan_to_num(np.asarray(ds[surf_var].values)).astype(np.float32)
        out["surface"] = sa
        a2 = sa.reshape(sa.shape[0], -1) if sa.ndim == 3 else sa
        print(f"  FastFuels surface '{surf_var}' {sa.shape}: CV {sa.std()/(sa.mean()+1e-9):.3f} (uniform per class)", flush=True)

    np.savez_compressed(os.path.join(PROC, "ff_osbs_3d.npz"), dz=1.0, dx=HRES,
                        x0=cx - AOI_HALF, y0=cy - AOI_HALF, epsg=epsg,
                        vars=json.dumps({k: list(v.shape) for k, v in out.items()}), **out)
    # NetCDF mirror (Option C interop) for the canopy field, if present
    if "canopy_bulk_density" in out:
        FuelVoxelGrid(bulk_density=out["canopy_bulk_density"], dz=1.0, dy=HRES, dx=HRES,
                      attrs={"scenario": "fastfuels_canopy3d", "source": "FastFuels v1 TreeMap-voxelized canopy",
                             "note": "FastFuels' genuinely-3D layer = trees; its surface layer is uniform per FBFM40 class"}
                      ).to_netcdf(os.path.join(PROC, "ff_osbs_canopy3d.nc"))
        print("  wrote data/processed/ff_osbs_3d.npz + ff_osbs_canopy3d.nc", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
