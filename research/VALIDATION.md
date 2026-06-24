# Validation Plan

> Validation and credibility are the most heavily weighted criteria in the 3D Surface Fuels & Vegetation Modeling Prize Challenge. This document lays out, before any model is built, exactly what we treat as truth, how we measure agreement against it, how we hold data out to avoid self-deception, and what sanity checks a skeptical judge should expect to see pass. The whole point of our entry is that FastFuels' surface fuels are *modeled* (a 30 m LANDFIRE Scott & Burgan FBFM40 raster expanded to a spatially uniform per-class loading, e.g. 0.717 kg/m² and 6.1 cm bed depth for one fuel class, which the FastFuels authors themselves note "does not capture the full complexity of surface fuel heterogeneity") whereas ours is *measured* from SAR + LiDAR. That claim only matters if we can prove it against independent ground truth ([Marcozzi et al. 2025, FastFuels](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)).

---

## What Counts as Truth for 3D Fuels

There is no single "truth" for a 1 m³ fuel voxel. Different reference methods measure different attributes at different scales, with different error structures. The central methodological problem is the **dimensional mismatch**: the historical gold standard for fuel *mass* is a 2D destructive clip plot integrated over ~1 m², while the challenge asks for a 3D *voxel* field at 1 m³ with per-layer bulk density. We therefore use a layered truth hierarchy rather than one reference, matching each measured property to the method that measures it best.

### 1. Destructive clip-plot harvest — gold standard for loading and live/dead fraction

Clip, oven-dry, and weigh all surface/understory biomass in fixed quadrats, separated by component (grass, litter, shrub, fine woody) and by live vs. dead. Directly yields **fuel loading (kg/m²)** and **live/dead fraction (%)** with essentially no model in between.

- **Pros.** Unambiguous; this is the attribute judges trust most, and the highest-value voxel property. It is what every remote-sensing fuel study ultimately calibrates against.
- **Cons.** It is 2D/vertically-integrated mass — a single clip plot does *not* resolve the vertical bulk-density profile of a voxel column unless it is sampled in vertical strata. It is destructive, sparse, and labor-intensive. There is a real temporal-offset risk: the clip date may not match the remote-sensing acquisition (e.g. a NEON flight year vs. a herbaceous harvest date), which we must document and, where possible, restrict to matched seasons.
- **Uncertainty.** Dominated by spatial sampling variance (surface fuels in longleaf systems are spatially independent beyond ~0.5 m — the "wildland fuel cell" — so a single quadrat is a noisy estimate of its surroundings; [Hiers et al. 2009](https://www.publish.csiro.au/wf/WF08084)) and by oven-drying / sorting error (small).

### 2. Terrestrial laser scanning (TLS) voxel grids — accepted reference for 3D structure

Co-register multi-scan TLS (thousands of pts/m²), clip to a plot radius, voxelize (2 cm is the established grain; the SERDP 3D Fuels work uses a 10×10×10 cm porosity index and 1 cm interpolation), and treat **occupied-voxel volume** and the vertical occupancy profile as the reference for **structure, depth, and the bulk-density distribution**.

- **Pros.** This is the only practical method that gives a true 3D occupancy/depth field aligned to the challenge's voxel target. The precedent is strong: TLS occupied volume (2 cm voxels, 15 m radius) explained **~85% of variance (R² ≈ 0.85) in destructively weighed surface/ground fuel mass at ~16% relative RMSE** ([Rowell et al. 2020, *Forest Ecology and Management*](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)). A separate longleaf study related occupied-voxel density to destructive bulk density by a logarithmic fit (adj. R² ≈ 0.32) before/after prescribed fire ([Rowell et al. 2020, treesearch 59822](https://research.fs.usda.gov/treesearch/59822)). An automated TLS pipeline now runs in <5 min/plot, making TLS a repeatable reference generator rather than a one-off ([USFS GTR, treesearch 67919](https://research.fs.usda.gov/treesearch/67919)).
- **Cons.** **Occlusion** is the central error source — in dense understory the scanner cannot see the lowest, densest fuel, so voxels are flagged occluded above ~80% occlusion and occupied volume is biased low near the ground. TLS is plot-scale (tens of m), not wall-to-wall, so it validates the *method* but not landscape generality on its own. Converting occupancy to kg/m³ still requires a destructive calibration subsample, so TLS is not truth-independent of clip plots — it is a structural bridge between clip mass and a 3D field.
- **Uncertainty.** ~16% relative RMSE on mass at plot scale ([Rowell et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)); occlusion bias grows with canopy/understory density; voxel-occupancy-vs-truth can be as weak as R² ≈ 0.33 in difficult conditions, which is itself a reason we add an independent penetrating sensor (SAR).

### 3. 3D stratified destructive sampling — best alignment to voxel layers

Harvest small columns (e.g. 0.5×0.5 m, the FERA 3D Fuels design) in 10 cm vertical strata to 1–2 m height, weighing by stratum and component. This yields **layer-resolved bulk density (kg/m³)** that lines up with voxel layers, plus live/dead by stratum. This is the cleanest reference for the per-voxel bulk-density requirement.

- **Pros.** Directly comparable to a 1 m voxel column; gives bulk density per layer, not just integrated mass.
- **Cons.** Extremely labor-intensive; very small footprint; few public per-stratum tables exist; requires upscaling to map.
- **Uncertainty.** Strong per-column fidelity but very sparse spatial coverage.

### 4. Planar-intersect (Brown's transects) — woody/coarse debris

Count and size woody debris crossing transect planes, converting to loading via standard volume/density equations. The standard, cheap reference for the **coarse and fine woody debris** component.

- **Pros.** Well-accepted, inexpensive, large effective sample length.
- **Cons.** Line-based, not volumetric; poor for grasses/litter; assumes regional density constants. Note that even good lidar+optical models fail almost entirely on coarse woody debris (effectively R² ≈ 0; [Hudak-style temperate study, *RSE* 2023](https://www.sciencedirect.com/science/article/abs/pii/S0034425723002626)), so we will treat the woody component cautiously and report it separately rather than blending it into a headline number.

### 5. Photoload / fuelbed photo-series — auxiliary densification only

Estimate loading by visual match to a calibrated photo series. Useful only to densify validation points and check spatial heterogeneity; **not** primary truth because of observer bias and coarseness. We will not use it as a headline reference for judges.

### Summary: which method validates which attribute

| Required voxel property | Primary reference | Secondary / cross-check |
|---|---|---|
| Fuel loading (kg/m²) | Destructive clip harvest | TLS occupied volume |
| Live/dead fraction (%) | Clip harvest (sorted) | NEON hyperspectral, field records |
| Bulk density (kg/m³) | 3D stratified harvest | TLS porosity/occupancy → calibration |
| Depth / height (m) | TLS vertical profile | Field fuelbed depth, TLS fuel-height models |
| Coarse/fine woody load | Planar-intersect transects | — |
| SAVR (m²/m³) | Species/fuel-type priors from field component data | — (not remotely sensed) |
| Patch/clump heterogeneity | TLS voxel statistics (variogram, CV) | High-res clip-plot spatial layout |

We are explicit that **SAVR and species are not delivered by SAR or LiDAR** — they come from field/fuel-type priors. We will not overclaim them as "measured."

---

## Our Validation Design

### Holdout strategy — spatial blocking, not random splits

Surface fuels are spatially autocorrelated at fine scales ([Hiers et al. 2009](https://www.publish.csiro.au/wf/WF08084)), so a random train/test split would leak information across the autocorrelation range and inflate scores. We use **spatially blocked cross-validation**: hold out whole plots (and, for the transfer test, whole sites) so that no test cell shares a neighborhood with a training cell. We report metrics on the held-out blocks only. Where a SAR feature has a 10 m native footprint, blocks are sized well above that footprint so SAR pixels are not shared across the split.

### Quantitative metrics

**(a) Fuel loading / bulk density — regression metrics.** On held-out plots we report, for loading (kg/m²) and for bulk density (kg/m³):
- **RMSE** and **relative RMSE** (RMSE / mean observed) — the headline number. The bar to beat is the TLS-vs-destructive benchmark of **~16% relative RMSE** ([Rowell et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)); we report ours against that bar honestly.
- **Bias** (mean signed error). Prior lidar-only surface-fuel work systematically *underpredicts at high loads* ([Kaibab Plateau study, RMSE 8–30 Mg/ha, underprediction at high loads](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)); we will report the residual-vs-observed plot specifically to check for this.
- **R²** (coefficient of determination). The relevant competitor ceilings to beat are documented and modest: lidar-only surface fuels R² ≈ 0.39–0.59 ([Kaibab](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)) and lidar+optical litter/fine-woody R² ≈ 0.27–0.41 ([RSE 2023](https://www.sciencedirect.com/science/article/abs/pii/S0034425723002626)). Our SAR+LiDAR fusion claim is credible only if we beat these on the same stratum.

**(b) Voxel structure — occupancy and profile agreement.** Where TLS reference voxels exist (RxCADRE/3D Fuels), we compare predicted vs. reference 1 m occupancy with:
- **Voxel occupancy IoU** (intersection-over-union of occupied voxels) — a direct 3D structural agreement metric.
- **Vertical-profile correlation** — Pearson/Spearman correlation between predicted and reference bulk-density-by-height profiles, per column. This tests whether we put fuel at the right *heights*, not just the right total.

**(c) Heterogeneity / clump structure — distributional metrics.** The single biggest qualitative failure of FastFuels' surface layer is *spatial uniformity within a fuel class*. Matching the mean is not enough; we must show we recover the variation that QUIC-Fire/FIRETEC actually consume (rate of spread is a nonlinear/power-law function of bulk density, so heterogeneity drives fire behavior; [Linn et al. 2020, QUIC-Fire](https://www.sciencedirect.com/science/article/abs/pii/S1364815219307388)). We therefore report, against TLS reference:
- **Coefficient of variation (CV)** of 1 m bulk density within a fuel class — compared head-to-head against the *single uniform value* FastFuels assigns to that class.
- **Variogram range** (the distance at which fuel becomes spatially independent) — the empirical target is ~0.5 m in longleaf ([Hiers et al. 2009](https://www.publish.csiro.au/wf/WF08084)); we report whether our predicted field reproduces a plausible autocorrelation range and patch size, drawing on the FastFuels layerset `random_clusters` parameters (`patch_size`, `patch_std_dev`).
- **Distribution agreement** (e.g. KS statistic / Earth Mover's distance) between predicted and reference loading distributions, not just their means.

### Benchmark against the incumbent — measured vs. modeled, head-to-head

For every footprint we also compute the **FastFuels SB40-lookup surface layer** and report (i) our error vs. ground truth and (ii) FastFuels' error vs. the *same* ground truth, plus the heterogeneity FastFuels structurally cannot produce (its within-class CV is 0 by construction). Because we deliver a FastFuels-compatible 1 m voxelized netCDF bulk-density array (challenge pathway 2) or a layerset GeoJSON (pathway 3), the comparison is apples-to-apples on identical truth.

### Cross-site / cross-ecosystem transfer test

Generality is a named judging axis. We hold out *entire ecosystems*: fit on a SE US longleaf/flatwoods site, test on a structurally different site (a western grassland and/or ponderosa), and report the per-ecosystem RMSE/bias/R² so the degradation under transfer is visible and honest. NEON's 81 sites and free national 3DEP/Sentinel-1 coverage make this feasible. We expect (and will state) that transfer degrades accuracy and that per-ecosystem refitting is the intended operational mode.

### Uncertainty quantification — per-voxel confidence

A credible product reports its own uncertainty. We attach **per-voxel prediction intervals** via quantile regression forests or ensemble spread, and validate that they are *calibrated* (i.e. ~90% of held-out truth falls inside the nominal 90% interval). This mirrors the standard set by spaceborne products that ship native per-estimate error: GEDI L4A carries a per-footprint prediction standard error and L4B a per-1 km-cell standard error via hybrid estimation ([GEDI L4A user guide, ORNL DAAC](https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density.html)). We will also flag voxels where the prediction leans on SAR rather than LiDAR returns (i.e. under dense canopy), since those carry the SAR resolution/saturation caveats below.

### Honest limitations we will state up front

- **SAR resolution and saturation.** Free spaceborne SAR is coarse relative to 1 m (Sentinel-1 ~10–20 m; NISAR ~3–10 m native), and backscatter saturates at low-to-moderate biomass and is *noisy at the <1 kg/m² ≈ 10 Mg/ha loads where surface fuels live* — well below the L-band HH saturation ceiling of ~105 Mg/ha ([L-band review, PMC11397620](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397620/)). Sentinel-1 C-band VV/VH has been *excluded* from final rangeland biomass models due to moisture/phenology confounding ([rangeland AGB study, PMC10682297](https://pmc.ncbi.nlm.nih.gov/articles/PMC10682297/)). We therefore frame SAR as a penetrating *covariate / occlusion-filler*, sharpened to 1 m by LiDAR where LiDAR exists — not as a standalone 1 m fuel sensor.
- **ALS near-ground weakness.** Airborne lidar PAD inversion validates well above 0.5 m but is weak at the ground (R² ≈ 0.4) due to occlusion — exactly where surface fuels are ([Martin-Ducup et al. 2025, lidarforfuel](https://hal.inrae.fr/hal-04908881v1)). This is the gap SAR is meant to partially fill, and we will quantify it.
- **GEDI is canopy, sparse, and 25 m** — a scaling/transfer baseline, not surface-fuel truth (surface/herb/shrub strata R² ≈ 0.15–0.46; [Leite et al. 2022](https://www.fs.usda.gov/rm/pubs_journals/2022/rmrs_2022_leite_r001.pdf)).

---

## Truth Datasets We Can Use Now

All datasets below are open and have working programmatic access as of mid-2026. They are chosen so that a remote-sensing **input** is paired with an **independent** field/structure reference.

### RxCADRE 2012 — Eglin AFB, FL (longleaf pine grass/shrub) — strongest single validation pair

Input and truth are co-located in the *same* plots, both public-domain, no login.
- **Input:** RxCADRE 2012 TLS point clouds (LAZ; UTM X/Y, normalized height, intensity) for the three Highly Instrumented Plots L1G (grass), L2G (grass/shrub), L2F (forest). DOI [10.2737/RDS-2023-0011](https://www.fs.usda.gov/rds/archive/catalog/RDS-2023-0011) ([AgData Commons mirror](https://agdatacommons.nal.usda.gov/articles/dataset/RxCADRE_2012_Terrestrial_laser_scan_TLS_point_cloud_data_for_Eglin_Air_Force_Base/27010462)).
- **Truth:** RxCADRE ground fuel measurements (pre/post-fire loading by fuelbed component, consumption, moisture) — RDS-2014-0028 / RDS-2014-0031 ([RDS-2014-0031](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2014-0031)). Reference magnitudes: forest units ~6.8 Mg/ha load, ~4.1 Mg/ha consumption; non-forest ~3.0 / ~2.2 Mg/ha. Maps to **loading, depth, moisture, live/dead**; the TLS gives the **voxel-IoU / vertical-profile** reference. There is also a validated TLS fuel-height model precedent here ([Rowell et al. 2015, IJWF](https://www.publish.csiro.au/wf/wf14170)).

### NEON Ordway-Swisher (OSBS), FL (longleaf/turkey-oak savanna) — best scalable demo pair

- **Input:** Discrete-return airborne LiDAR DP1.30003.001 (1×1 km tiles; ~4–8 pts/m²) and CHM DP3.30015.001 (1 m raster) for the canopy mask. Both via `neonutilities` one-liner `by_tile_aop(...)` ([DP1.30003.001](https://data.neonscience.org/data-products/DP1.30003.001), [neonutilities](https://github.com/NEONScience/NEON-utilities-python)).
- **Truth:** Herbaceous clip harvest DP1.10023.001 (destructive dry mass → kg/m², maps to **loading + live/dead**), litterfall/fine woody DP1.10033.001, coarse downed wood DP1.10014.001 (maps to **woody load** via the planar method). Co-sampled on the same plots; downloadable via `load_by_product(...)` ([DP1.10023.001](https://data.neonscience.org/data-products/DP1.10023.001)).

### SERDP RC19-1064 "3D Fuels" — canonical voxel→bulk-density calibration

The authoritative hierarchical TLS calibration set: 18 intensively-sampled sites across SE and western US, a 10×10×10 cm **porosity index directly related to bulk density and scaled to QUIC-Fire/FIRETEC/FDS**, with shrub-height QSM models at R² 0.98–0.99 ([SERDP RC1064 Final Report, Aug 2024](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf); FERA 3D Fuels design at [depts.washington.edu/fera/3dfuels](https://depts.washington.edu/fera/3dfuels/)). This supplies our **bulk-density (kg/m³)** reference and the cross-ecosystem transfer plots.

### Supporting input layers (free, programmatic)

- **Sentinel-1 C-band SAR** (VV/VH GRD, ~10 m) via `asf_search` ([ASF docs](https://docs.asf.alaska.edu/datasets/using_ASF_data/)) — the SAR covariate.
- **NISAR L+S-band** (free, launched 30 Jul 2025, 12-day repeat, ~3–10 m native) — the strongest free penetrating SAR going forward ([NISAR, eoPortal](https://www.eoportal.org/satellite-missions/nisar)). We note its post-2025 archive is short, so multi-temporal history may need ALOS-2 PALSAR-2 as a fallback.
- **USGS 3DEP ALS** via OpenTopography PDAL/EPT ([OT_3DEP_Workflows](https://github.com/OpenTopography/OT_3DEP_Workflows)) — national-coverage input for generality.
- **GEDI L2A/L2B/L4A** via `earthaccess` ([GEDI-Data-Resources](https://github.com/nasa/GEDI-Data-Resources)) — UQ baseline and scaling cross-check only.
- **LANDFIRE FBFM40** (30 m, layer `240FBFM40`) via the LFPS REST API ([LFPS User Guide](https://lfps.usgs.gov/LFProductsServiceUserGuide.pdf)) — the **modeled baseline to beat**, identical to FastFuels' surface source.

### Dataset → metric mapping

| Dataset | Provides | Validates metric |
|---|---|---|
| RxCADRE 2012 TLS (RDS-2023-0011) | 3D occupancy reference | Voxel IoU, vertical-profile r, heterogeneity CV/variogram |
| RxCADRE ground fuels (RDS-2014-0031) | Clip loading, moisture | Loading RMSE/bias/R², live/dead |
| NEON DP1.10023 / .10033 / .10014 | Clip / litter / CWD mass | Loading RMSE/bias/R², woody component |
| NEON DP1.30003 / DP3.30015 | ALS input + canopy mask | (input, not truth) |
| SERDP 3D Fuels porosity | Bulk density by stratum | Bulk-density RMSE; cross-ecosystem transfer |
| LANDFIRE FBFM40 | Modeled per-class loading | Head-to-head baseline |
| GEDI L4A/L4B | Per-estimate standard error | UQ calibration reference |

---

## The "Smell Test"

Before reporting any model metric, the product must pass the sanity checks a fuels scientist or judge will apply on sight. We will publish these as an explicit pre-flight checklist.

### Mass totals in plausible per-ecosystem ranges

- **Longleaf pine / flatwoods surface load.** Forested RxCADRE units averaged ~6.8 Mg/ha (≈ 0.68 kg/m²), non-forest ~3.0 Mg/ha (≈ 0.30 kg/m²) ([RxCADRE ground fuels](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2014-0031)). A predicted 1 m surface map whose plot means land in roughly **0.2–1.0 kg/m²** for this ecosystem is plausible; values an order of magnitude off are a red flag.
- **Rangeland/grass AGB** is typically ~0.21–0.47 kg/m² ([rangeland study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10682297/)) — a useful bound for the grassland transfer site.
- **Surface ≫ canopy mass.** Surface fuel load can be ~12× available canopy fuel (82.7 vs 6.9 Mg/ha at Kaibab; [PMC9361251](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)). If our surface totals come out trivially small relative to canopy, something is wrong — and it reinforces *why* the surface layer dominates fire-model error.

### Depth vs. known fuelbed depth

Predicted fuel-bed depth should track documented values: the Anderson FM9 long-needle-litter benchmark FastFuels uses is **6.1 cm**; grass/shrub beds run higher. Litter depths of meters, or grass beds of millimeters, fail the test. FastFuels' own uniform value (0.717 kg/m², 6.1 cm) is the literal reference point.

### Bulk density consistency

Bulk density must equal loading ÷ depth and stay within physically sensible ranges for the fuel type (litter and grass beds are low-density, packed duff higher). We cross-check the derived kg/m³ against the SERDP 3D Fuels porosity-derived bulk densities for the matching fuel type ([SERDP RC1064](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf)).

### Heterogeneity is non-zero and spatially structured

Our 1 m field must show **non-trivial within-class variance** (the whole thesis) but also realistic spatial structure — an autocorrelation range near the ~0.5 m wildland-fuel-cell scale, not white noise and not the flat single value FastFuels produces ([Hiers et al. 2009](https://www.publish.csiro.au/wf/WF08084)). A map that is either perfectly uniform (we failed to beat FastFuels) or spatially random (we are fitting noise) fails.

### Live/dead and seasonal sanity

Live/dead fractions must be in [0, 100]% and track season/curing. Because radar backscatter responds to vegetation water content, a multi-temporal SAR moisture/curing signal should move in the expected direction with phenology; if it does not, the SAR feature is suspect.

### Mass conservation under the FastFuels prior (perturbation mode)

If we deliver the "learned heterogeneity field that perturbs FastFuels' uniform layer" variant, the landscape mean of our field must reconcile with the FastFuels/LANDFIRE class mean (we add structure, we do not silently rescale total mass). We will report the before/after landscape-mean loading to show conservation.

### Residual diagnostics

We inspect residual-vs-observed plots for the known **underprediction-at-high-load** failure mode of lidar surface-fuel models ([Kaibab](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)), residual maps for spatial structure (leftover autocorrelation = a missing covariate), and uncertainty-interval coverage (≈90% empirical coverage for nominal 90% intervals). Any systematic pattern is reported, not hidden.

---

### One-line summary for judges

We validate a *measured* surface-fuel field against *independent* destructive and TLS reference data on the exact attributes the challenge requires — loading, bulk density, depth, live/dead, and heterogeneity — using spatially-blocked holdouts, calibrated per-voxel uncertainty, a cross-ecosystem transfer test, and a head-to-head benchmark against FastFuels' modeled SB40 surface layer on identical ground truth. The bar we hold ourselves to is the ~16% relative-RMSE TLS-vs-destructive benchmark and beating the documented lidar-only surface-fuel R² ceilings of 0.27–0.59.
