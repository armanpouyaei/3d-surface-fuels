"""Build the SUBMISSION DELIVERABLES over OSBS — FastFuels-compatible 1 m voxels.

Produces, on one aligned ~510 m AOI:
  1. osbs_measured_1m.nc   — MEASURED 3D bulk density (kg/m³) at 1 m from 3DEP
                             LiDAR (FastFuels "Option C" NetCDF). The product where
                             airborne LiDAR exists.
  2. osbs_uniform_1m.nc    — the FastFuels/LANDFIRE-style UNIFORM surface layer for
                             the same AOI (one column profile everywhere; CV → 0).
  3. osbs_generalized_1m.nc — the GENERALIZED product: the end-to-end spaceborne→10 m
                             structure (build_pipeline_osbs.py) disaggregated to 1 m
                             voxels with the measured vertical profile. The product
                             where only spaceborne data exist (the global path).
  + figures/deliverable_osbs.png — property maps (load / depth / bulk density / occupancy)
    measured vs uniform vs generalized, with the heterogeneity + pattern-agreement summary.

All three share the SAME AOI/footprint so the dashboard can show them head-to-head.
Light: one point-cloud load + binning; NetCDFs ~8 MB each (data/ is gitignored).

Usage:  python scripts/build_deliverable_osbs.py   (needs osbs_3dep_2018.laz + pipeline_osbs.npz)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, GeoRef, lidar, downscale as ds, metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
LAZ = os.path.join(ROOT, "data", "raw", "osbs_3dep_2018.laz")
SRC_EPSG, TGT_EPSG = 6438, 32617
NZ, DZ = 8, 1.0           # 0–8 m surface + understory + low ladder, 1 m³ voxels
AOI_CELLS10 = 51          # 51 × 10 m = 510 m AOI (aligned to the pipeline 10 m grid)


def clean_grid(bd, x0, y0, attrs):
    """A FuelVoxelGrid for export — bulk density only, clean attrs (no bulky extras)."""
    return FuelVoxelGrid(bulk_density=bd.astype(np.float32), dz=DZ, dy=DZ, dx=DZ,
                         georef=GeoRef(f"EPSG:{TGT_EPSG}", x0, y0, DZ), attrs=attrs)


PROP_NOTE = ("bulk_density, fuelbed depth and load are MEASURED from LiDAR; live/dead "
             "fraction, SAVR and fuel moisture are not LiDAR-observable and are left to the "
             f"challenge sentinel {1.23456} (provide via species/FCCS lookup downstream).")


def main():
    if not os.path.exists(LAZ):
        sys.exit(f"Missing {LAZ}")
    if not os.path.exists(os.path.join(PROC, "pipeline_osbs.npz")):
        sys.exit("Missing pipeline_osbs.npz — run scripts/build_pipeline_osbs.py first.")
    pipe = np.load(os.path.join(PROC, "pipeline_osbs.npz"), allow_pickle=True)
    e2e10, res10 = pipe["e2e"], float(pipe["res10"])      # (150,150) @10 m, row0=south
    x0p, y0p = float(pipe["x0"]), float(pipe["y0"])
    ny10, nx10 = e2e10.shape

    # centered AOI aligned to the 10 m grid -> exact nesting with the 1 m voxel grid
    o = (ny10 - AOI_CELLS10) // 2
    x0a, y0a = x0p + o * res10, y0p + o * res10
    size = AOI_CELLS10 * res10                            # 510 m
    cx, cy = x0a + size / 2, y0a + size / 2
    print(f"AOI {size:.0f}×{size:.0f} m at 1 m → {int(size)}×{int(size)}×{NZ} voxels; "
          f"origin ({x0a:.0f},{y0a:.0f}) EPSG:{TGT_EPSG}")

    print("Loading OSBS 3DEP 2018 + voxelizing measured 1 m³ bulk density...")
    pc = lidar.load_points(LAZ, src_epsg=SRC_EPSG, target_epsg=TGT_EPSG, z_unit="ft_us")
    mvox = lidar.voxelize(pc, center=(cx, cy), size=size, nz=NZ, dz=DZ)   # x0 = cx-size/2 = x0a
    measured = clean_grid(mvox.bulk_density, x0a, y0a, {
        "scenario": "measured", "site": "NEON OSBS (Ordway-Swisher), FL",
        "source": "USGS 3DEP LiDAR FL_Peninsular_2018 (~17 pts/m²)",
        "ecosystem": "longleaf pine sandhill / savanna",
        "submission_pathway": "FastFuels Option C — 3D voxelized bulk density, 1 m",
        "properties_note": PROP_NOTE})

    # FastFuels-style uniform layer for the same AOI
    uniform = lidar.uniform_baseline(measured)
    uniform.attrs = {"scenario": "fastfuels_uniform",
                     "note": "LANDFIRE/SB40-style single-class uniform surface layer (CV→0)",
                     **uniform.attrs}

    # generalized: spaceborne→10 m end-to-end, cropped to the AOI, disaggregated to 1 m,
    # distributed vertically by the measured mean profile (spaceborne cannot resolve the
    # vertical understory shape — borrow it; an honest, documented assumption).
    crop = e2e10[o:o + AOI_CELLS10, o:o + AOI_CELLS10]
    factor = int(round(res10 / DZ))                       # 10
    load_gen = ds.upsample(crop, factor, (int(size), int(size)))
    prof = measured.vertical_profile()
    prof = prof / (prof.sum() + 1e-9)                     # normalized vertical shape
    bd_gen = (load_gen[None, :, :] * prof[:, None, None] / DZ).astype(np.float32)
    generalized = clean_grid(bd_gen, x0a, y0a, {
        "scenario": "generalized_spaceborne", "site": "NEON OSBS, FL",
        "source": "AlphaEarth + Sentinel-1 + terrain → 30 m → 10 m (end-to-end), disaggregated to 1 m",
        "submission_pathway": "FastFuels Option C — 3D voxelized bulk density, 1 m",
        "vertical_note": "vertical profile borrowed from the measured mean (spaceborne resolves load, not vertical shape)",
        "properties_note": PROP_NOTE})

    # write NetCDFs
    os.makedirs(PROC, exist_ok=True)
    for name, g in [("osbs_measured_1m.nc", measured), ("osbs_uniform_1m.nc", uniform),
                    ("osbs_generalized_1m.nc", generalized)]:
        g.to_netcdf(os.path.join(PROC, name))
        print(f"  wrote data/processed/{name}  {g.shape}")

    # ── validation / smell test ─────────────────────────────────────────────────
    ml, ul, gl = measured.fuel_load(), uniform.fuel_load(), generalized.fuel_load()
    # rescale generalized to the measured mean for a fair PATTERN comparison
    glr = gl * (ml.mean() / (gl.mean() + 1e-9))
    r_pat = float(np.corrcoef(glr.ravel(), ml.ravel())[0, 1])
    print("\nValidation (1 m AOI):")
    print(f"  measured     load mean {ml.mean():.3f} kg/m²  CV {ml.std()/ml.mean():.2f}  "
          f"depth {measured.fuelbed_depth().mean():.2f} m  occ {measured.occupancy().mean()*100:.0f}%")
    print(f"  uniform      load mean {ul.mean():.3f} kg/m²  CV {ul.std()/ul.mean():.3f}  (FastFuels-style, flat)")
    print(f"  generalized  load mean {gl.mean():.3f} kg/m²  CV {gl.std()/gl.mean():.2f}  "
          f"| pattern corr vs measured r={r_pat:+.2f}")

    # ── property-maps figure ────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    flip = np.flipud
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.6))
    vmax = float(np.percentile(ml, 99))
    def show(a, arr, title, cmap="YlOrBr", vmn=0, vmx=vmax):
        im = a.imshow(flip(arr), cmap=cmap, vmin=vmn, vmax=vmx, aspect="equal")
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([]); return im
    im0 = show(ax[0, 0], ml, "MEASURED load (kg/m²)")
    show(ax[0, 1], ul, "FastFuels uniform load")
    show(ax[0, 2], glr, "GENERALIZED load (spaceborne→1 m)")
    imd = show(ax[0, 3], measured.fuelbed_depth(), "Measured fuelbed depth (m)", "viridis", 0,
               float(np.percentile(measured.fuelbed_depth(), 99)))
    plt.colorbar(im0, ax=ax[0, :3], fraction=0.012, pad=0.01, label="kg/m²")
    plt.colorbar(imd, ax=ax[0, 3], fraction=0.025, pad=0.02, label="m")
    # bottom: bulk-density top-down (max over z), occupancy, vertical profiles, CV bars
    show(ax[1, 0], measured.bulk_density.max(0), "Measured bulk density (max over z)", "magma", 0,
         float(np.percentile(measured.bulk_density.max(0), 99)))
    show(ax[1, 1], measured.occupancy().mean(0), "Measured voxel occupancy (frac of column)", "Greens", 0, 1)
    zc = measured.heights()
    ax[1, 2].plot(measured.vertical_profile(), zc, "-o", color="#31a354", label="measured")
    ax[1, 2].plot(generalized.vertical_profile() * (measured.vertical_profile().sum() /
                  (generalized.vertical_profile().sum() + 1e-9)), zc, "-s", color="#2c7fb8", label="generalized")
    ax[1, 2].plot(uniform.vertical_profile(), zc, "--", color="#d95f0e", label="uniform")
    ax[1, 2].set_xlabel("bulk density (kg/m³)"); ax[1, 2].set_ylabel("height (m)")
    ax[1, 2].set_title("Vertical profile"); ax[1, 2].legend(fontsize=8)
    cvs = [ml.std()/ml.mean(), gl.std()/gl.mean(), ul.std()/(ul.mean()+1e-9)]
    ax[1, 3].bar(["measured", "generalized", "uniform"], cvs, color=["#31a354", "#2c7fb8", "#d95f0e"])
    ax[1, 3].set_ylabel("heterogeneity (CV)")
    ax[1, 3].set_title(f"Heterogeneity (FastFuels≈0)\ngeneralized↔measured pattern r={r_pat:+.2f}")
    fig.suptitle("OSBS submission deliverables — measured vs FastFuels-uniform vs generalized (1 m³, FastFuels Option C)",
                 fontsize=13, y=1.0)
    fp = os.path.join(ROOT, "figures", "deliverable_osbs.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=110, bbox_inches="tight")
    print(f"  saved {fp}")
    print("\nDeliverables ready: 3D voxel NetCDFs (Option C) + property maps. "
          "Dashboard reads osbs_*_1m.nc for the 3D view and property panels.")


if __name__ == "__main__":
    main()
