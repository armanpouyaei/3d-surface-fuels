# 3D Surface Fuels & Vegetation Modeling Prize Challenge — Phase-1 Submission

**A measured, field-calibrated 1 m surface-fuel product that beats FastFuels' uniform layer — from free spaceborne data, anywhere.**

Phase-1 deadline 2026-07-20. All results below are from scripts in `scripts/` on **real** data with
spatially-blocked cross-validation; every number is reproducible. Details: [research/RESULTS.md](research/RESULTS.md),
[research/VALIDATION.md](research/VALIDATION.md), [research/RESEARCH.md](research/RESEARCH.md).

---

## 1. One-paragraph method

FastFuels paints surface fuel as **one number per 30 m LANDFIRE fuel-model class** — uniform within the
class, by construction. We instead **measure understory structure at 1 m**: from LiDAR we compute
**vertical occupancy** (the fraction of 0.15–4 m height-bins that contain returns — a density-robust
structure metric), voxelize it to 1 m³ bulk density, and **field-calibrate it to surface fuel load
against RxCADRE destructive clip plots** (occupancy → total load, slope 1.33 kg/m², R² 0.93). Where
airborne LiDAR is absent, a **per-pixel gradient-boosting model** predicts that same occupancy from
**globally-free spaceborne inputs** — AlphaEarth embeddings + Sentinel-1 C-band + ALOS PALSAR L-band +
global canopy height — with **conformal uncertainty** and an **out-of-distribution flag**, then
disaggregates to a FastFuels-compatible 1 m³ NetCDF. Simple, explainable, and every layer is measured or
field-anchored — no hallucinated structure.

**Differentiator vs the host team's ForestGen3D and de Conto (2025):** they *generate/predict canopy
structure*; we deliver an **independent physical measurement** of the **surface/understory** layer,
calibrated to **destructive ground truth**, with honest uncertainty.

---

## 2. Deliverables (this submission)

| # | Deliverable | Where |
|---|---|---|
| Data product | 1 m³ voxel NetCDF, FastFuels "Option C" (OSBS: measured, uniform-baseline, generalized) | `data/processed/osbs_*_1m.nc` via `scripts/build_deliverable_osbs.py` |
| Fuel properties | bulk_density (kg/m³), fuel_load (kg/m²), percent_cover (%), SAVR (1/m), live_fraction, **dead + live fuel moisture (%)**, heat of combustion (kJ/kg) | in the NetCDF |
| P3 landscape metrics | avg patch size (m), heterogeneity CV below 2 m | NetCDF attrs (`p3_*`) |
| Geospatial locational data | AOI boundary polygon, WGS84 + native CRS | `data/processed/osbs_boundary.geojson` |
| Property maps + 3D viz | property-map figure + interactive 3D | `figures/deliverable_osbs.png`, dashboard |
| **Validation documentation** | methodology, truth data, blocked-CV, metrics, field calibration, limitations | [research/RESULTS.md](research/RESULTS.md), [research/VALIDATION.md](research/VALIDATION.md) |
| **Python ingestion/visualization tool** | one command → summary + smell test + maps + 3D | `python scripts/read_deliverable.py <file.nc>` |
| Global "generate anywhere" demo | pick any AOI → predicted surface load + uncertainty + OOD, FastFuels side-by-side | `streamlit run dashboard/app.py` |

Empty cells use the challenge sentinel `1.23456`. We satisfy **both** submission pathways: geo-referenced
1 m arrays (Option 1) *and* FastFuels-compatible NetCDF (Option 2).

---

## 3. Evidence, mapped to the Phase-1 judging criteria

### (1) Utility of the methodology
Physics-based fire models (QUIC-Fire, FIRETEC, FDS) consume **sub-class surface heterogeneity**
nonlinearly — exactly what FastFuels' uniform layer discards. Head-to-head against the real FastFuels
method (LANDFIRE FBFM40 → SB40 load lookup) on the OSBS 1 m product:

| product | overall R² | within-block R² (sub-30 m) | heterogeneity CV |
|---|---|---|---|
| FastFuels uniform | −0.00 | **0.00 (by construction)** | 0.00 |
| **ours (end-to-end)** | **0.664** | **0.274** | **0.73** (truth 0.89) |

Our within-block R² **0.27** *is* the heterogeneity FastFuels structurally cannot represent. The output
is a drop-in FastFuels-compatible product, now in **field-calibrated kg/m²** (§validation).

### (2) Generality of approach (ecosystem applicability)
The identical pipeline runs across **15 ecosystems on 4 continents** with no code changes — US savanna,
desert, temperate/Sierra conifer, deciduous, wetland; European temperate + hemiboreal; **equatorial
rainforest (Amazon + Borneo)**. Cross-ecosystem leave-one-site-out (the honest generality test):

- **Between-biome / absolute level: solved** — GLOBAL R² **0.49**, between-biome R² **0.81** (placing an
  unseen biome at the right absolute level; AlphaEarth classifies biome at ~100%).
- **Within-biome fine structure, unseen forest: works** (median within-R² +0.11; within-Spearman ~0.3–0.6, mean ~0.4 across forested sites).
- A conservative **per-biome-radius OOD flag** marks locations unlike *any* training ecosystem, so the
  product is *grounded where the world resembles training, honestly flagged elsewhere*
  (`scripts/check_ood_global.py`, `scripts/global_coverage_table.py`).

### (3) Data acquisition challenges & cost
**100% free / open, no field campaign required at inference.** Inputs: USGS 3DEP LiDAR (ungated TNM/rockyweb),
AlphaEarth (free Source-Coop S3 mirror), Sentinel-1 RTC + ALOS PALSAR + Copernicus DEM (Microsoft
Planetary Computer, ungated), Meta/WRI global canopy height (open AWS), ESA WorldCover. Truth for
calibration/validation: RxCADRE destructive plots (USFS RDS, open) + NEON. No proprietary data, no keys
for the core pipeline. The global model needs **only spaceborne inputs at inference** — LiDAR is used to
*train*, not to deploy.

### (4) Clarity of methodology
Every step compresses to one sentence and every number is script-reproducible. We deliberately use a
**per-pixel gradient-boosting model** — and *proved* it's the right choice at this scale via a faithful
head-to-head vs the de Conto (2025) fully-convolutional EfficientNetV2 paradigm on the same target/folds:
**ours R² 0.628 (85% conformal coverage) vs the CNN −0.147 (broken 14% coverage)** — the CNN's
spatial-context advantage needs continental training our tiled design supports, but is the wrong tool for
the AOI-scale task the challenge poses (`scripts/deconto_headtohead.py`).

### (5) Credibility of validation approach *(top-weighted — our strongest axis)*
- **Spatially-blocked cross-validation** everywhere (no spatial leakage); **leave-one-site-out** for
  cross-ecosystem generality.
- **Calibrated uncertainty**: conformal prediction intervals (85% empirical coverage) + a per-cell OOD score.
- **Field calibration to destructive truth**: occupancy → RxCADRE total surface load, **R² 0.93** across 9
  Eglin burn blocks (grass 0.21 → forest 1.12 kg/m²) — the absolute kg/m² scale is anchored to clip plots,
  not guessed (`scripts/calibrate_load_field.py`). Independent NEON field check included and reported honestly.
- **Honest negative results reported**: cross-ecosystem *within*-biome R² is hard; NEON herb clips are the
  wrong stratum for our metric (R² 0.02) — documented, not hidden.

### (6) Relevance to surface & understory fuels
Our target is explicitly the **0.15–4 m understory** stratum (vertical occupancy → surface load),
calibrated to RxCADRE **surface** clip plots — distinct from canopy-structure indices (de Conto's WSCI)
and from FastFuels' canopy voxels. SAR/canopy-height features let us infer understory where optical
saturates. This is the surface/understory layer the challenge targets.

### (7) Practical scalability
Global by design: free spaceborne inputs, **on-demand per-AOI inference** with a canonical-tile disk cache,
**memory-safe by construction** (10 m inference, hard AOI cap, voxel-budgeted 1 m export). Coverage extends
**tile-by-tile** — each new open ALS tile brings its biome in-distribution (demonstrated: adding Amazon +
Borneo moved the tropics from OOD to in-distribution).

---

## 4. Limitations (stated plainly)
- **Absolute-load calibration is single-ecosystem** (Eglin/RxCADRE, R² 0.93 across a wide range). Strong,
  but multi-site destructive plots would refine per-biome transfer; forests read slightly low (occupancy
  under-weights heavy litter), deserts slightly high.
- **Within-biome fine structure in a *brand-new* biome** is partially solved (forests) — the genuine
  sensing limit (optical/C-band don't see under closed canopy); L-band + canopy height help, GEDI is sparse.
- **Fuel moisture (P2)** is populated but coarse: dead FM is ERA5-driven (Simard EMC; ERA5 ~9–31 km,
  so near-uniform at AOI scale), live FM is a Sentinel-2 NDVI-scaled proxy (uncalibrated LFMC). Most P3
  metrics are now included (avg patch size, <2 m heterogeneity, heat of combustion); species mix is out
  of scope for Phase 1.
- The OOD flag is conservative by design — it warns rather than silently extrapolates.

## 5. Reproducibility
`pip install -r requirements.txt`, then: `build_deliverable_osbs.py` (product), `read_deliverable.py`
(ingest/viz), `build_pipeline_osbs.py [--site …]` (end-to-end + generality), `calibrate_load_field.py`
(RxCADRE calibration), `compare_load_vs_fastfuels.py`, `global_coverage_table.py`, `deconto_headtohead.py`,
`streamlit run dashboard/app.py` (interactive). See [TODO.md](TODO.md) for remaining pre-deadline items
(fuel moisture via ERA5, cheap P3 props, final packaging).

## 6. Eligibility & IP
Team eligibility (18+, U.S./NATO, non-federal, no federal funds) per the challenge terms. We retain IP;
the Government/partners receive permanent access per the rules. No protective markings applied.
