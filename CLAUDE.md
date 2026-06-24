# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A competition entry for the **3D Surface Fuels & Vegetation Modeling Prize Challenge**
(Central Florida Tech Grove / NAWCTSD / SERDP-ESTCP). **Goal: win it.** Phase-1
deadline **2026-07-20**; prizes $40k / $25k / $20.1k.

We produce **meter-scale (1 m³ voxel) 3D maps of surface/understory fuels** (grass,
shrub, litter, woody debris) and propose a **better-than-FastFuels** method using
**SAR + LiDAR**. Deliverables: georeferenced 3D fuel arrays (FastFuels-compatible
NetCDF), property maps + 3D viz, **validation docs (judged most heavily)**, and a
Python visualization/ingestion tool.

## The thesis (don't lose sight of this)

FastFuels' *canopy* fuels are genuinely 3D (FIA/TreeMap imputation → crown
voxelization), but its **surface** fuels are a **LANDFIRE SB40/FCCS 30 m categorical
lookup** → spatially uniform per fuel-model class (e.g. FM9 = 0.717 kg/m², 6.1 cm
everywhere). The FastFuels paper admits this. Real surface fuels are heterogeneous
below ~0.5 m and fire models consume that nonlinearly. **We measure surface fuels at
1 m from LiDAR understory returns + SAR canopy-penetration, validate against
destructive/TLS truth, and benchmark head-to-head against FastFuels' uniform layer.**

Differentiator vs the host team's own **ForestGen3D** (diffusion that *hallucinates*
sub-canopy structure from ALS): we add an **independent physical measurement** (radar),
not a generative guess.

## Working principles (the user's explicit constraints)

- **Simple & explainable** > clever. Every method must compress to one paragraph.
- **New & scientific** — defensible novelty, grounded in cited literature.
- **Validation-first** — it is the top-weighted judging criterion. Lead with loading
  R²/RMSE and heterogeneity (CV); report 3D metrics honestly (pixelwise RMSE can
  favor a mean-reverting uniform baseline — say so).
- **Free/open data only** (Sentinel-1, NISAR, 3DEP, NEON, RxCADRE, LANDFIRE).
- **Build simple first, then layer on.** v0 (synthetic) already runs; don't rewrite it.

## Repository map

| Path | Purpose |
|---|---|
| `research/RESEARCH.md` | Full research: challenge, FastFuels internals, the gap, LiDAR/SAR, datasets, ~50 cited refs |
| `research/VALIDATION.md` | Validation strategy, truth datasets, metrics, smell tests |
| `IDEAS.md` | Candidate approaches (A–F), ranked; recommended MVP; week-by-week roadmap |
| `src/surface_fuels/voxel.py` | `FuelVoxelGrid` — 1 m³ bulk-density grid + NetCDF I/O + derived products |
| `src/surface_fuels/synthetic.py` | `make_demo_scenario()` → truth / fastfuels / ours grids |
| `src/surface_fuels/metrics.py` | `compare()` and individual validation metrics |
| `src/surface_fuels/lidar.py` | real LAZ → `FuelVoxelGrid` (load_points/reproject, ground DTM, voxelize, find_aoi, canopy_cover_grid, uniform_baseline) |
| `src/surface_fuels/sar.py` | real Sentinel-1 RTC features (VV/VH/ratio/RVI) from Planetary Computer + real fused blend |
| `src/surface_fuels/fusion.py` | canopy-weighted SAR+LiDAR blend + spatially-blocked CV + stratified metrics |
| `src/surface_fuels/basemap.py` | real Esri World Imagery basemap for a grid's footprint |
| `dashboard/app.py` | Streamlit dashboard — **Synthetic** + **SAR+LiDAR fusion** + **Real Eglin** modes |
| `scripts/make_demo_data.py` | synthetic NetCDFs + validation report |
| `scripts/run_fusion_demo.py` | synthetic fusion blocked-CV validation table |
| `scripts/download_eglin_lidar.py` | fetch real USGS 3DEP LiDAR over Eglin (TNM Access → rockyweb, ungated) |
| `scripts/build_eglin.py` | voxelize the LAZ → eglin_measured/uniform.nc + eglin_basemap.png (auto-open AOI) |
| `scripts/build_sar_eglin.py` | fetch real Sentinel-1 + canopy → eglin_sar.npz (fused product) |
| `data/{raw,interim,processed}/` | Data (gitignored) |

## Commands

```bash
source .venv/bin/activate                  # Python 3.9 venv (system python3)
python scripts/make_demo_data.py           # synthetic grids + validation report
python scripts/run_fusion_demo.py          # SAR+LiDAR fusion blocked-CV table
python scripts/download_eglin_lidar.py     # real Eglin 3DEP LiDAR (~33 MB → data/raw/)
python scripts/build_eglin.py              # voxelize + fetch real basemap → data/processed/
python scripts/build_sar_eglin.py          # real Sentinel-1 + fused product → data/processed/
streamlit run dashboard/app.py             # dashboard (http://localhost:8501)
```

No test suite yet; validate changes by running `make_demo_data.py` (ours ≫ FastFuels
baseline) and `build_eglin.py` (measured CV ≈ 0.46 vs uniform 0; canopy↔imagery corr
≈ −0.49 brightness / +0.54 greenness confirms same-spot + orientation). The dashboard 3D
volume is WebGL-heavy — `_coarsen()` in app.py caps display cells (~45k) for responsiveness.

**Dashboard map-panel gotcha:** in `load_maps`, all top-down panels share matched axes
(`matches='x'/'y'`) so zoom/pan is synchronized. `go.Image` forces the shared y-axis to
*reversed* (row 0 at top), so every array (load heatmaps AND the RGB) is `np.flipud`-ed to
row 0 = north → north-up + aligned. Load arrays are row 0 = south; the Esri basemap and the
LiDAR canopy share that orientation (verified by correlation), so they line up.

## Conventions

- Voxel arrays are indexed **`[z, y, x]`** (z = height up); voxel size 1 m default.
- Bulk density in **kg/m³**; fuel loading in **kg/m²** = `Σ_z ρ · dz`.
- NetCDF output follows FastFuels **Option C** (3D voxelized bulk density at 1 m).
- No-fuel cells: mass 0, other properties default to the challenge sentinel `1.23456`.
- Keep code dependency-light (numpy/scipy/xarray); add heavy geo deps only in the
  real-data scripts (commented in `requirements.txt`).

## Key reference facts (so you don't re-research)

- **FastFuels v2 API:** https://api-v2-prod-782971006568.us-west1.run.app/ · docs
  https://docs.fastfuels.silvxlabs.com/v2-beta/ · core repo `silvxlabs/fastfuels-core`.
  Surface = `/grids/fbfm40/landfire` → `/grids/lookup/fbfm40` (uniform per class).
  Heterogeneous surface only via user-supplied **layerset** GeoJSON.
- **Quick-start data:** RxCADRE 2012 Eglin TLS (`RDS-2023-0011`) + ground fuels
  (`RDS-2014-0031`) — input+truth co-located, public domain. NEON OSBS for scalable
  airborne (`neonutilities`). SERDP RC19-1064 "3D Fuels" = TLS↔bulk-density calibration.
- **Validation bar:** TLS occupied-volume ↔ destructive mass ≈ R² 0.85, ~16% rRMSE.
  LiDAR-only surface-fuel R² ceiling ≈ 0.27–0.59 (the bar to beat).
- **SAR honest limits:** Sentinel-1 ~10 m, saturates/noisy below ~10 Mg/ha; role is
  occlusion-filler + moisture/curing proxy, sharpened to 1 m by LiDAR — not a 1 m sensor.

## Real-data access notes (learned in Week 1)

- **RxCADRE TLS (RDS-2023-0011) + clip plots (RDS-2014-0031)** are the ideal co-located
  input+truth pair, but AgData Commons and FS RDS are **bot-gated** (HTTP 202/403) and
  not reliably scriptable. Download manually from the catalog pages into `data/raw/` to
  plug into `lidar.voxelize()`.
- **USGS 3DEP via TNM Access API is ungated** and scriptable — used as the real-LiDAR
  on-ramp. The Eglin tile is `FL_OKALOOSACO_2007` (EPSG:2238, ftUS, ~4.3 pts/m², legacy
  low density). `lidar.load_points` reprojects 2238→UTM 16N and converts ftUS→m.
- **Esri World Imagery export** (no key) is the satellite basemap source (`basemap.py`).
- **Sentinel-1 RTC via Microsoft Planetary Computer is ungated** — `planetary_computer.sign_inplace`
  + `pystac_client` + windowed `rasterio` reproject. RTC is gamma0 VV/VH at 10 m already in
  EPSG:32616 over Eglin, so it reads straight into the grid. Used by `sar.py`. (ASF/asf_search
  and OpenTopography both need credentials/keys — avoid.)

## Status & next step

- ✅ **Week 0:** synthetic pipeline, voxel model, metrics, dashboard.
- ✅ **Week 1:** real Eglin 3DEP LiDAR → 1 m measured surface-fuel grid; real satellite
  basemap; dashboard Real mode (measured CV 0.46 / Moran's I 0.44 vs uniform 0).
- ✅ **Week 2:** SAR+LiDAR canopy-weighted fusion. Synthetic blocked-CV: fusion R² 0.71 vs
  LiDAR-only 0.51 / SAR-only 0.37 / uniform 0; under canopy 0.50 vs LiDAR −0.01. Real
  Sentinel-1 RTC over Eglin (VH/VV + fused product) in the dashboard.
- ▶️ **Next:** (a) RxCADRE clip plots (manual) for real R²/RMSE + absolute calibration;
  (b) Week 3 — NEON OSBS (recent LiDAR + S1, no temporal gap) wall-to-wall; FastFuels
  head-to-head; export FastFuels Option C/D.

When adding real data, mirror the `FuelVoxelGrid` interface so dashboard/metrics work unchanged.
