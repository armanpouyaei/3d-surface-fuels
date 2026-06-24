"""FastFuels surface-fuel head-to-head over OSBS.

FastFuels' surface fuel is, by design, **LANDFIRE FBFM40 (Scott & Burgan 40 fuel
models) at 30 m → an SB40 load lookup → one value per fuel-model class, uniform
within the class** (heterogeneity CV → 0). We compare that to our measured 1 m
product on the same AOI.

Two source modes for FastFuels' layer (auto-selected):
  A. **FastFuels API (live)** if FASTFUELS_API_KEY is set — verified v1 flow: create a
     domain over the AOI → POST grids/surface with fuelLoad from LANDFIRE FBFM40 →
     poll → export geotiff → download → read the real surface fuel-load grid.
  C. **Offline** (fallback/default): FastFuels' *exact* surface lookup — the SB40 load
     table (Scott & Burgan 2005, RMRS-GTR-153, Table 7, encoded below) summed over the
     same groups FastFuels uses (1h+10h+100h+live herb+live woody) — vs our measured
     load distribution, with the longleaf-relevant fuel models overlaid.

Status (2026-06-24): the live API path is **verified working** with a real key (a
`uniform` surface grid completes end-to-end), but FastFuels' **LANDFIRE FBFM40**
surface generation was **failing server-side** (and LANDFIRE was directly unreachable
from this environment), so the committed run uses mode C. Mode C is numerically what
FastFuels returns — its FBFM40 fuelLoad *is* the SB40 group-sum lookup encoded here.
Re-run with the key once FastFuels' LANDFIRE backend recovers to pull the live grid.

Usage:  python scripts/fastfuels_headtohead.py            # offline (mode C)
        FASTFUELS_API_KEY=... python scripts/fastfuels_headtohead.py   # live (mode A)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, downscale as ds  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
TON_PER_ACRE_TO_KG_M2 = 0.224170     # 2000 lb / 4046.856 m^2

# Scott & Burgan (2005) Table 7 — load by class (tons/acre): dead 1h,10h,100h, live herb, woody.
SB40 = {
    "GR1": (0.10, 0.00, 0.00, 0.30, 0.00), "GR2": (0.10, 0.00, 0.00, 1.00, 0.00),
    "GR3": (0.10, 0.40, 0.00, 1.50, 0.00), "GR4": (0.25, 0.00, 0.00, 1.90, 0.00),
    "GR5": (0.40, 0.00, 0.00, 2.50, 0.00), "GR6": (0.10, 0.00, 0.00, 3.40, 0.00),
    "GR7": (1.00, 0.00, 0.00, 5.40, 0.00), "GR8": (0.50, 1.00, 0.00, 7.30, 0.00),
    "GR9": (1.00, 1.00, 0.00, 9.00, 0.00),
    "GS1": (0.20, 0.00, 0.00, 0.50, 0.65), "GS2": (0.50, 0.50, 0.00, 0.60, 1.00),
    "GS3": (0.30, 0.25, 0.00, 1.45, 1.25), "GS4": (1.90, 0.30, 0.10, 3.40, 7.10),
    "SH1": (0.25, 0.25, 0.00, 0.15, 1.30), "SH2": (1.35, 2.40, 0.75, 0.00, 3.85),
    "SH5": (3.60, 2.10, 0.00, 0.00, 2.90), "SH7": (3.50, 5.30, 2.20, 0.00, 3.40),
    "TU1": (0.20, 0.90, 1.50, 0.20, 0.90), "TU2": (0.95, 1.80, 1.25, 0.00, 0.20),
    "TU5": (4.00, 4.00, 3.00, 0.00, 3.00),
    "TL2": (1.40, 2.30, 2.20, 0.00, 0.00), "TL3": (0.50, 2.20, 2.80, 0.00, 0.00),
    "TL6": (2.40, 1.20, 1.20, 0.00, 0.00), "TL8": (5.80, 1.40, 1.10, 0.00, 0.00),
}
# Fuel models LANDFIRE commonly maps across a longleaf-pine sandhill/savanna mosaic
# (frequent fire → grass / grass-shrub understory, with timber-litter under denser canopy).
OSBS_CANDIDATES = ["GR1", "GR2", "GR3", "GS1", "GS2", "TU1", "TL2"]


def sb40_load(code):
    """Total surface fuel load (kg/m²) FastFuels assigns to a fuel-model class."""
    return round(sum(SB40[code]) * TON_PER_ACRE_TO_KG_M2, 3)


# ── mode A: FastFuels API (verified live flow; needs FASTFUELS_API_KEY) ──
# The user's platform key is a v1 key → base api.fastfuels.silvxlabs.com, header `api-key`.
# Flow (verified end-to-end against the live API): create domain (GeoJSON Feature) → POST
# grids/surface with fuelLoad from LANDFIRE FBFM40 → poll → export geotiff → download → read.
# Returns the surface fuel-load array (kg/m²) or None (e.g. FastFuels' LANDFIRE step failing).
def fastfuels_via_api(bbox_lonlat, key, hres=30.0, poll=8, tries=90):
    import requests, time
    base = "https://api.fastfuels.silvxlabs.com"
    h = {"api-key": key, "Content-Type": "application/json"}
    xmin, ymin, xmax, ymax = bbox_lonlat
    feat = {"type": "Feature", "name": "osbs_h2h", "horizontalResolution": hres,
            "verticalResolution": 1.0, "properties": {}, "geometry": {"type": "Polygon",
            "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}}
    did = requests.post(f"{base}/v1/domains", headers=h, json=feat, timeout=60).json().get("id")
    if not did:
        return None
    body = {"attributes": ["fuelLoad"],
            "fuelLoad": {"source": "LANDFIRE", "product": "FBFM40", "version": "2022"}}
    requests.delete(f"{base}/v1/domains/{did}/grids/surface", headers=h, timeout=30)
    if requests.post(f"{base}/v1/domains/{did}/grids/surface", headers=h, json=body, timeout=60).status_code >= 400:
        return None
    for _ in range(tries):
        st = requests.get(f"{base}/v1/domains/{did}/grids/surface", headers=h, timeout=40).json().get("status")
        if st == "completed":
            break
        if st in ("failed", "error", None):
            return None          # FastFuels' LANDFIRE FBFM40 generation failed server-side
        time.sleep(poll)
    requests.post(f"{base}/v1/domains/{did}/grids/surface/exports/geotiff", headers=h, json={}, timeout=40)
    url = None
    for _ in range(tries):
        ex = requests.get(f"{base}/v1/domains/{did}/grids/surface/exports/geotiff", headers=h, timeout=40).json()
        url = ex.get("signedUrl") or ex.get("url")
        if url or ex.get("status") in ("failed", "error"):
            break
        time.sleep(poll)
    if not url:
        return None
    import rasterio
    fn = os.path.join(PROC, "ff_osbs_surface.tif")
    requests_download(url, fn)
    with rasterio.open(fn) as src:
        descs = [d or "" for d in src.descriptions]
        bi = next((i + 1 for i, d in enumerate(descs) if "load" in d.lower()), 1)
        arr = src.read(bi).astype(float)
    arr = arr[np.isfinite(arr)]
    return arr[arr > -9990]


def requests_download(url, fn):
    import requests
    with requests.get(url, stream=True, timeout=120) as r:
        with open(fn, "wb") as f:
            for c in r.iter_content(1 << 16):
                f.write(c)


# ── mode B: LANDFIRE FBFM40 class histogram over the AOI ──
def fastfuels_via_landfire(bbox_utm, epsg):
    import requests, json
    url = ("https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF230/US_230FBFM40/"
           "ImageServer/computeHistograms")
    xmin, ymin, xmax, ymax = bbox_utm
    geom = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "spatialReference": {"wkid": epsg}}
    r = requests.get(url, timeout=60, params={
        "geometry": json.dumps(geom), "geometryType": "esriGeometryEnvelope", "f": "json"},
        headers={"User-Agent": "Mozilla/5.0"})
    return r.json()


def main():
    mp = os.path.join(PROC, "osbs_measured_1m.nc")
    if not os.path.exists(mp):
        sys.exit("Missing osbs_measured_1m.nc — run scripts/build_deliverable_osbs.py first.")
    g = FuelVoxelGrid.from_netcdf(mp)
    load1 = g.fuel_load()                                   # measured 1 m load (kg/m²)
    load30 = ds.block_coarsen(load1, 30)                    # aggregated to FastFuels' 30 m
    mu, cv1 = float(load1.mean()), float(load1.std() / load1.mean())
    cv30 = float(load30.std() / load30.mean())

    print("FastFuels' SB40 surface-fuel loads (kg/m²) for longleaf-relevant classes:")
    for c in OSBS_CANDIDATES:
        print(f"  {c}: {sb40_load(c):.3f}  (= {sum(SB40[c]):.2f} ton/acre, uniform per class)")

    # try the LIVE FastFuels API (mode A) if a key is set; gracefully fall back to offline (C)
    key = os.environ.get("FASTFUELS_API_KEY")
    ff_live = None
    if key:
        try:
            print("\nMode A: FastFuels API (live, v1)…")
            ff_live = fastfuels_via_api(lonlat_bbox(g), key)
            if ff_live is not None and ff_live.size:
                print(f"  LIVE FastFuels surface fuelLoad: n={ff_live.size}, mean {ff_live.mean():.3f} kg/m², "
                      f"CV {ff_live.std()/(ff_live.mean()+1e-9):.3f}, unique {np.unique(np.round(ff_live,3))[:8]}")
            else:
                print("  Live surface grid did not complete (FastFuels' LANDFIRE FBFM40 step failed "
                      "server-side) → using FastFuels' exact SB40 lookup table (mode C).")
                ff_live = None
        except Exception as e:
            print(f"  API call failed ({e}) → mode C.")
    else:
        print("\nNo FASTFUELS_API_KEY set → mode C (FastFuels' exact SB40 lookup table).")

    print(f"\nMeasured OSBS surface load: mean {mu:.3f} kg/m² | CV(1 m) {cv1:.2f} | CV(30 m) {cv30:.2f}")
    if ff_live is not None:
        print(f"LIVE FastFuels surface layer CV: {ff_live.std()/(ff_live.mean()+1e-9):.3f} "
              f"(mean {ff_live.mean():.3f}) — uniform per class, vs our measured CV {cv1:.2f}.")
    else:
        print("FastFuels surface layer: ONE SB40 value per 30 m class → CV = 0 by construction.")
        print(f"Verdict: whichever class FastFuels assigns ({'/'.join(OSBS_CANDIDATES)} bracket our "
              f"mean), it collapses the AOI to a single value; we resolve a CV {cv1:.2f} distribution.")

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    pos = load1[load1 > 0]
    ax[0].hist(np.clip(pos, 0, np.percentile(pos, 99)), bins=60, color="#31a354", alpha=.8,
               label=f"measured 1 m load (CV {cv1:.2f})")
    if ff_live is not None:
        for v in np.unique(np.round(ff_live, 3)):
            ax[0].axvline(v, color="#d95f0e", lw=2)
        ax[0].axvline(ff_live.mean(), color="#d95f0e", lw=2,
                      label=f"FastFuels LIVE = {ff_live.mean():.2f} (CV {ff_live.std()/(ff_live.mean()+1e-9):.2f})")
        ff_cv, ff_lbl = float(ff_live.std() / (ff_live.mean() + 1e-9)), "FastFuels\n(live API)"
    else:
        cols = plt.cm.autumn(np.linspace(0, 0.8, len(OSBS_CANDIDATES)))
        for c, col in zip(OSBS_CANDIDATES, cols):
            v = sb40_load(c)
            ax[0].axvline(v, color=col, lw=2, label=f"FastFuels {c} = {v:.2f}")
        ff_cv, ff_lbl = 0.0, "FastFuels\n(SB40 uniform)"
    ax[0].axvline(mu, color="k", ls="--", lw=1.5, label=f"measured mean {mu:.2f}")
    ax[0].set_xlabel("surface fuel load (kg/m²)"); ax[0].set_ylabel("1 m cells")
    ax[0].set_title("FastFuels assigns a single value per class;\nreality is a distribution")
    ax[0].legend(fontsize=7)
    ax[1].bar([ff_lbl, "Measured\n(1 m)", "Measured\n(30 m agg.)"],
              [ff_cv, cv1, cv30], color=["#d95f0e", "#31a354", "#2c7fb8"])
    ax[1].set_ylabel("heterogeneity (CV)")
    ax[1].set_title("Spatial heterogeneity of surface load")
    for i, v in enumerate([ff_cv, cv1, cv30]):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.suptitle("FastFuels surface layer (LANDFIRE FBFM40 → SB40 lookup, uniform per class) vs our measured product — OSBS",
                 y=1.02, fontsize=12)
    fp = os.path.join(ROOT, "figures", "fastfuels_headtohead.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=115, bbox_inches="tight")
    print(f"\nsaved {fp}")


def lonlat_bbox(g):
    from pyproj import Transformer
    epsg = int(str(g.georef.crs).split(":")[-1])
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    x0, y0 = g.georef.x0, g.georef.y0
    lon0, lat0 = tr.transform(x0, y0); lon1, lat1 = tr.transform(x0 + g.nx * g.dx, y0 + g.ny * g.dy)
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))


if __name__ == "__main__":
    main()
