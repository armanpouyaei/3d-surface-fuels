# 3D Surface Fuels — a measured alternative to FastFuels' modeled surface layer

A submission-in-progress for the [3D Surface Fuels & Vegetation Modeling Prize Challenge](https://centralfloridatechgrove.org/surface-fuels-prize-challenge/)
(Central Florida Tech Grove / NAWCTSD / SERDP-ESTCP). **Phase-1 deadline: 2026-07-20.**

## The one-sentence thesis

> FastFuels paints surface fuel as **one number per 30 m fuel-model class** (e.g. Anderson FM9 = 0.717 kg/m², 6.1 cm — uniform everywhere). We **measure it at 1 m** by voxelizing LiDAR understory returns, filled by **SAR where the canopy occludes LiDAR**, calibrated to destructive/TLS truth — and deliver it as a drop-in FastFuels-compatible product.

FastFuels' own paper admits its surface layer is modeled, not measured, and "does not capture the full complexity of surface fuel heterogeneity." Real surface fuels reorganize at sub-metre scale, and physics-based fire models (QUIC-Fire, FIRETEC, FDS) consume that heterogeneity nonlinearly. That gap is what we exploit. See [research/RESEARCH.md](research/RESEARCH.md).

## What's here now

The dashboard has **two modes**:

**1. Synthetic longleaf demo** — the full pipeline on a synthetic longleaf/wiregrass scene: *truth* (heterogeneous) vs *FastFuels baseline* (uniform per class) vs *ours* (recovered), with the complete validation report. Seed 42: **ours** load R² ≈ 0.74, CV ratio ≈ 0.99; **FastFuels baseline** R² ≈ 0, CV ratio 0 (all heterogeneity lost).

**2. SAR + LiDAR fusion (synthetic)** *(Week 2)* — the validated method. Spatially-blocked cross-validation shows **fusion R² 0.71** beats LiDAR-only (0.51), SAR-only (0.37), and FastFuels-uniform (0). Critically, **under canopy** fusion scores **0.50 vs LiDAR-only −0.01** — LiDAR is occluded there and the canopy-penetrating SAR fills the gap. `fused = (1−canopy)·LiDAR + canopy·SAR`.

**3. Real — Eglin 3DEP LiDAR** *(Week 1+2)* — a **measured 1 m surface-fuel grid from 10 M real USGS 3DEP LiDAR points over Eglin AFB**, over **real Esri World Imagery**, vs the FastFuels-style uniform layer (measured CV 0.46, Moran's I 0.44 vs uniform 0). Plus **real Sentinel-1 SAR** (10 RTC acquisitions, Apr–Jun 2026, from Planetary Computer): the VH/VV cross-pol ratio and a real SAR+LiDAR fused product over the same footprint. Same-spot + orientation verified by LiDAR-canopy↔imagery correlation.

Modules: [`voxel.py`](src/surface_fuels/voxel.py) (1 m³ grids + FastFuels NetCDF I/O), [`synthetic.py`](src/surface_fuels/synthetic.py), [`metrics.py`](src/surface_fuels/metrics.py), [`lidar.py`](src/surface_fuels/lidar.py) (LAZ → voxel grid), [`sar.py`](src/surface_fuels/sar.py) (Sentinel-1 features), [`fusion.py`](src/surface_fuels/fusion.py) (blend + blocked CV), [`downscale.py`](src/surface_fuels/downscale.py) (30 m→1 m, mass-conserving), [`basemap.py`](src/surface_fuels/basemap.py) (satellite tiles), [`dashboard/app.py`](dashboard/app.py).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_demo_data.py          # synthetic grids + validation report
python scripts/run_fusion_demo.py         # SAR+LiDAR fusion: blocked-CV validation table
python scripts/run_downscale_demo.py      # 30m→1m downscaler POC: blocked-CV table
python scripts/download_eglin_lidar.py    # real USGS 3DEP LiDAR over Eglin (~33 MB)
python scripts/build_eglin.py             # voxelize it + fetch real satellite basemap
python scripts/build_sar_eglin.py         # fetch real Sentinel-1 + build fused product
python scripts/build_downscale_eglin.py   # real 30m→1m downscale over Eglin
streamlit run dashboard/app.py            # launch the 3D dashboard
```

## Deliverable format & ingestion tool

Each deliverable is a **FastFuels-compatible Option-C NetCDF** of **1 m³ voxels** plus a sibling
**AOI-boundary GeoJSON** (WGS84 + native CRS — the challenge's required geospatial locational data).
The NetCDF carries, per the challenge's fuel-property list:

| variable | dims | units | tier |
|---|---|---|---|
| `bulk_density` | (z,y,x) | kg/m³ | **P1** cell bulk density |
| `fuel_load` | (y,x) | kg/m² | **P1** fuel load |
| `percent_cover` | (y,x) | % | **P1** percent cover |
| `savr` | (z,y,x) | 1/m | **P2** surface-area-to-volume |
| `live_fraction` | (z,y,x) | – | **P2** live vs. dead |

Empty cells use the challenge sentinel `1.23456`. CRS/origin/resolution are in the file attributes.

**One-command ingestion + visualization tool** (the required Python tool — needs only numpy/xarray/matplotlib):

```bash
python scripts/read_deliverable.py data/processed/osbs_measured_1m.nc
```

It prints a property summary + a validation smell-test (mean load, heterogeneity CV, fuelbed depth,
occupancy), reads the AOI boundary, and writes `<name>_maps.png` (fuel-property maps) and
`<name>_3d.png` (3D voxel view). Build the OSBS deliverable itself with
`python scripts/build_deliverable_osbs.py`.

## Repository layout

```
research/RESEARCH.md     Challenge, how FastFuels works, the gap, LiDAR/SAR, datasets, references
research/VALIDATION.md   Validation strategy (the most heavily weighted judging criterion)
IDEAS.md                 Candidate approaches, the recommended MVP, and the build roadmap
src/surface_fuels/       voxel model · synthetic generator · validation metrics
dashboard/app.py         Streamlit 3D viewer + truth comparison
scripts/                 make_demo_data.py and (soon) real-data download/voxelize scripts
data/                    raw / interim / processed (gitignored)
CLAUDE.md                Guide for working in this repo with Claude Code
```

## Roadmap (today → deadline)

| Week | Focus |
|---|---|
| 0 ✅ | Synthetic pipeline, voxel model, metrics, dashboard |
| 1 ✅ | Real Eglin LiDAR (USGS 3DEP) → 1 m bulk-density grid; real satellite basemap; measured-vs-uniform on real data |
| 2 ✅ | SAR+LiDAR fusion (canopy-weighted blend); blocked-CV validation (fusion 0.71 vs LiDAR 0.51); real Sentinel-1 over Eglin |
| 3 | **Go global (two-stage):** **(3a)** build the global **30 m product** from spaceborne (GEDI+S1+S2), then **(3b)** the **30 m→1 m downscaler** consumes it. Downscaler POC done (within-block R² 0→0.31). |
| 3b+ | Swap downscaler regressor for **AlphaEarth/Clay embeddings + UNet**; cross-ecosystem transfer test |
| 4 | RxCADRE clip plots (absolute calibration); FastFuels head-to-head; export Option C/D; package submission |

**Architecture — global by design (two stages):** a coarse **30 m surface-fuel product** from globally-free spaceborne sensors (Stage 1) is sharpened to **1 m** by a mass-conserving **downscaler** using globally-free 10 m covariates (Stage 2). Both are *trained over the US* (3DEP gives wall-to-wall targets) and *applied globally* (every input is a global sensor; only the target is US-limited). **Stage 1 is the prerequisite — build it first.** See [research/DOWNSCALING.md](research/DOWNSCALING.md) and [TODO.md](TODO.md).

Full plan and rubric mapping in [IDEAS.md](IDEAS.md).
