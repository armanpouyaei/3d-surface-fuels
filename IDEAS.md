# IDEAS.md — Strategy & Brainstorm: A Measured 3D Surface-Fuel Product to Beat FastFuels

> **Thesis (one sentence):** FastFuels paints surface/understory fuels as one number per 30 m fuel-model class; we *measure* surface fuels at meter scale by voxelizing LiDAR understory returns (filled by SAR where canopy occludes LiDAR), calibrated to destructive/TLS truth, and deliver it as a drop-in FastFuels-compatible product.

This document captures the design principles, the candidate methods we evaluated, the single recommended MVP, and a week-by-week build roadmap to the **July 20, 2026** deadline. It is grounded entirely in the consolidated research findings; where evidence is thin or uncertain, that is flagged inline.

> See [research/RESEARCH.md](research/RESEARCH.md) for the full evidence base and [research/VALIDATION.md](research/VALIDATION.md) for the validation design. The v0 synthetic pipeline that exercises all of this lives in [`src/surface_fuels/`](src/surface_fuels/) and [`dashboard/app.py`](dashboard/app.py).

---

## The gap we are attacking (why this can win)

The challenge weights **validation/credibility** most heavily, then utility, generality, data cost, clarity, relevance to surface/understory fuels, and scalability. Our entire strategy is built to maximize exactly those axes.

FastFuels' canopy product is genuinely 3D and measured-ish (FIA + TreeMap imputation → inhomogeneous Poisson tree placement → crown-profile voxelization with NSVB allometry, per-voxel bulk density/SAVR/moisture at 2 m × 2 m × 1 m). Its **surface** product is the documented weak link:

- The default surface layer comes straight from **LANDFIRE FBFM40 (Scott & Burgan, 46 classes) at 30 m**, converted to fuel parameters by a **static per-class lookup table** — so loading (kg/m²), bulk density, depth, and SAVR are **spatially uniform within each fuel-model class** and constant across every 30 m pixel of that class ([FastFuels API OpenAPI schema](https://api-v2-prod-782971006568.us-west1.run.app/openapi.json)).
- The FastFuels paper itself states the surface layer is modeled, not measured — e.g. an FDS demo modeling surface fuels as a *uniform* Anderson FM9 layer (**0.717 kg/m², 6.1 cm depth**), explicitly noting it "does not capture the full complexity of surface fuel heterogeneity (Vakili et al., 2016)" ([Marcozzi et al. 2025, RMRS PDF](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)).
- The **only** route to heterogeneous surface fuels in FastFuels is a **user-supplied "layerset"** — the user must already *know* loading, percent cover, patch size, and moisture; FastFuels supplies none of it ([fastfuels-core/layersets.py](https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/layersets.py)).

This matters physically: longleaf-pine work shows surface fuels become spatially independent beyond **~0.5 m** ("wildland fuel cell") and fire behavior/effects correlate with structure at **<0.25 m²** grain ([Hiers et al. 2009](https://www.publish.csiro.au/wf/WF08084); [Loudermilk et al. 2012](https://research.fs.usda.gov/treesearch/46764)). Physics-based models (QUIC-Fire, FIRETEC, FDS) consume **per-voxel bulk density** directly, and rate of spread is a power-law function of it ([Linn et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S1364815219307388)). Feeding them a uniform per-class value discards the variability they exist to resolve.

**The opening:** a measured, spatially heterogeneous 1 m surface-fuel layer that ingests cleanly into FastFuels (as a 1 m voxelized netCDF bulk-density array, pathway 2; or a statistical layerset GeoJSON, pathway 3) directly improves the most heavily weighted judging axes.

---

## Design Principles

1. **Simple & explainable.** The method must compress to one paragraph a fire scientist will nod at: *"LiDAR returns below ~2 m measure the surface fuel layer directly; we voxelize them and calibrate occupied volume to clip-plot mass; where the canopy hides the ground, free radar backscatter fills in."* No black-box hallucination as the load-bearing step.
2. **Novel but scientifically grounded.** Novelty comes from *measuring* the under-canopy zone (an independent radar observation) rather than inferring it from overstory proxies — the documented failure mode of all prior LiDAR-only surface-fuel work (R² ceiling ~0.27–0.59).
3. **Validation-first.** Every output property is benchmarked against independent destructive/TLS truth with reported R²/RMSE/bias, plus a head-to-head against FastFuels' uniform per-class value. Validation is the top-weighted rubric criterion; we design backwards from it.
4. **Scalable & generalizable.** Core inputs (USGS 3DEP / NEON ALS, Sentinel-1, NISAR, GEDI) are **free, national/global, and repeatable**. The pipeline re-fits per ecosystem; the code is identical across sites.
5. **Sensor-agnostic, anchored on free/open data.** LiDAR is the workhorse; SAR is the occlusion-filler and moisture/curing signal. We deliberately avoid paid commercial X-band SAR (Capella/ICEYE/Umbra: finest resolution but worst penetration, hurts data-cost score and senses only canopy tops).
6. **FastFuels-compatible by construction.** Outputs target the named submission pathways (1 m netCDF bulk density, or layerset GeoJSON) so utility/relevance is automatic and validation is apples-to-apples with FastFuels' own output grids.
7. **Honest about limits.** State plainly where each sensor fails (SAR saturation below ~10 Mg/ha; Sentinel-1 native 10 m vs the 1 m target; ALS occlusion near the ground). Credibility is built by not overclaiming.

---

## Candidate Approaches

Six candidates were evaluated. Each is described in plain English with data needs, novelty, 4-week feasibility, validation credibility, and how it beats FastFuels.

### A. ALS understory-return voxelization → calibrated 1 m bulk density
**Plain English.** Take free airborne LiDAR (USGS 3DEP / NEON), height-normalize it to the ground, then look only at the returns in the near-ground band (~0.15–2 m). Voxelize them at 1 m (and finer, 0.25 m, for the surface layer), compute per-voxel occupied-volume / return-density, and fit a simple regression mapping that to destructively measured surface-fuel loading (kg/m²) and depth, deriving bulk density = load/depth. This is the directly-measured surface layer FastFuels lacks.
**Data needed.** USGS 3DEP or NEON `DP1.30003.001` ALS; co-located clip-plot biomass (NEON `DP1.10023.001`, RxCADRE `RDS-2014-0031`) for calibration. All free.
**Novelty.** Moderate. The TLS→voxel→bulk-density chain is established (Rowell/Loudermilk, SERDP RC19-1064; [final report](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf)); applying it to *standard public ALS* as a heterogeneous surface layer is the incremental contribution.
**Feasibility (4 wk).** High. All data free and scriptable (PDAL); model is one regression.
**Validation credibility.** High — directly against clip plots; benchmark vs FastFuels uniform value.
**Beats FastFuels.** Replaces the uniform 30 m per-class value with a measured 1 m heterogeneity field. Honest weakness: ALS under-samples the lowest fuels under closed canopy (occlusion; near-ground R² ~0.4 in [Martin-Ducup et al. 2025](https://hal.inrae.fr/hal-04908881v1)).

### B. SAR + LiDAR fusion to fill canopy occlusion (the strongest novelty bet)
**Plain English.** Where the canopy is open, ALS understory returns measure the surface directly (as in A). Where the canopy occludes ALS, use **SAR backscatter** — which *penetrates* the canopy — as the predictor. L-band cross-pol (HV) / the HV-HH (or VH-VV) polarization ratio is empirically, monotonically correlated with understory density once canopy cover drops below a threshold ([ALOS-2 PALSAR-2 study, 2024](https://www.tandfonline.com/doi/abs/10.1080/15481603.2024.2360771)). Train one model (gradient-boosted / small fusion head) with **canopy cover as an explicit interaction term**, so it leans on SAR exactly where LiDAR goes blind. Output a heterogeneous 1 m surface-fuel field.
**Data needed.** Sentinel-1 GRD (free, 10 m, dual-pol, 6–12 day revisit, via `asf_search`); optionally NISAR L+S band (free since launch 30 Jul 2025, 24 cm L-band penetrates canopy, ~3–10 m native) or ALOS-2 L-band; ALS; TLS/clip-plot truth.
**Novelty.** High. Prior surface-fuel work stacks LiDAR + *optical* and still fails on under-canopy litter (R² 0.27–0.41; [RSE 2023](https://www.sciencedirect.com/science/article/abs/pii/S0034425723002626)). Using **radar penetration as an independent measurement** of the occluded zone is new and physically grounded.
**Feasibility (4 wk).** Medium-high. SAR features are simple; the resolution-mismatch story (10 m radar prior sharpened to 1 m by ALS) must be framed honestly.
**Validation credibility.** High — same TLS/clip truth, with explicit comparison to the documented LiDAR-only R² ceiling.
**Beats FastFuels.** Adds a *measured* under-canopy signal FastFuels assigns from a lookup, and beats the LiDAR-only ceiling on the exact stratum the challenge cares about.

### C. Learned heterogeneity field that DOWNSCALES FastFuels' uniform surface layer
**Plain English.** Keep FastFuels' per-class **mean** loading as a prior (so the landscape mean stays correct and trusted), then learn a spatial *residual* field from ALS understory metrics + SAR + terrain + texture/semivariogram descriptors. The output preserves the right mean but injects realistic patch/clump structure (patch size, variance, autocorrelation) that QUIC-Fire/FIRETEC need. Validate the *heterogeneity statistics* (variogram range, CV), not just the mean.
**Data needed.** FastFuels API output as baseline; ALS; Sentinel-1; TLS for heterogeneity-stat validation.
**Novelty.** Medium. Framing ("predict heterogeneity, not the mean") is clever and non-threatening to judges from the FastFuels community.
**Feasibility (4 wk).** High — builds on A/B with FastFuels as a free prior.
**Validation credibility.** Medium — validating heterogeneity metrics is less standard than validating loading, slightly harder to communicate.
**Beats FastFuels.** Attacks its exact weakness (uniformity) while staying compatible. More incremental than B.

### D. GEDI + Sentinel-1 spaceborne surface-fuel regression
**Plain English.** Use spaceborne GEDI waveform metrics + Sentinel-1 SAR in an ML regression to map surface/understory fuel proxies wall-to-wall and globally, calibrated by TLS allometry. The de Conto et al. 2025 GEDI+SAR EfficientNetV2 template hits global R²=0.82 for structural complexity at 25 m ([arXiv 2510.06299](https://arxiv.org/abs/2510.06299)).
**Data needed.** GEDI L2A/L2B (`earthaccess`), Sentinel-1, TLS allometry. All free.
**Novelty.** Medium. Fusion machinery is mature; surface-fuel target is the twist.
**Feasibility (4 wk).** Medium. GEDI is sparse 25 m footprints — far from 1 m — and surface accuracy is poor (R² 0.15–0.46 for lower strata; [Leite et al. 2022](https://www.fs.usda.gov/rm/pubs_journals/2022/rmrs_2022_leite_r001.pdf)).
**Validation credibility.** Medium. Best as a *scalability/generality* supporting layer, not the meter-scale core.
**Beats FastFuels.** On global generality and free data cost, not on meter-scale fidelity.

### E. Photogrammetry / UAV-SfM voxel occupancy (calibration backbone)
**Plain English.** Use UAV structure-from-motion or UAV-LiDAR voxel occupancy (10 cm voxels) on a few plots to derive surface fuel volume → bulk density/SAV, providing dense local truth where TLS is unavailable.
**Data needed.** UAV imagery/LiDAR on demo plots; clip plots for calibration.
**Novelty.** Low — established technique.
**Feasibility (4 wk).** Medium — requires field/flight data we likely won't collect in time.
**Validation credibility.** Medium — occlusion limits occupancy-vs-truth (R² ~0.33 reported; [bioRxiv 2019](https://www.biorxiv.org/content/10.1101/771469.full.pdf)).
**Beats FastFuels.** Only as a ground-truth densifier; not a standalone product.

### F. ALS Beer-Lambert PAD inversion (physics-based, landscape-scale)
**Plain English.** Invert the ALS point cloud to a vertical Plant Area Density profile via the Beer-Lambert light-extinction law, then convert PAD to a vertical bulk-density profile using species traits — integrating to layered load, bulk density, CBH/CBD per column. Open R package `lidarforfuel` ([Martin-Ducup et al. 2025](https://hal.inrae.fr/hal-04908881v1)).
**Data needed.** ALS with below-canopy returns; species/trait lookup (PAD→biomass). Free.
**Novelty.** Medium — established for canopy; near-ground application is the contribution.
**Feasibility (4 wk).** High — existing R package, one physical law + traits.
**Validation credibility.** Medium-high overall (R² 0.42 load, 0.68 CBD) but **weak exactly at the ground** (R² ~0.4, occlusion) where surface fuels live.
**Beats FastFuels.** Wall-to-wall, field-independent, explainable; but the surface stratum is its weakest, which is our whole point.

### Ranked comparison

Scores are 1–5 (5 = best), assigned from the findings. **Composite** is an unweighted mean of the four columns, shown only as a rough ordering aid; in practice we over-weight **validation** since it is the top rubric criterion.

| # | Approach | Novelty | Feasibility (4 wk) | Validation credibility | Scalability | Composite |
|---|----------|:-------:|:------------------:|:----------------------:|:-----------:|:---------:|
| **B** | **SAR + LiDAR occlusion-fill fusion** | **5** | **4** | **4** | **4** | **4.25** |
| A | ALS understory-return voxelization | 3 | 5 | 5 | 3 | 4.00 |
| C | Heterogeneity-field downscaler of FastFuels | 4 | 4 | 3 | 4 | 3.75 |
| F | ALS Beer-Lambert PAD inversion | 3 | 4 | 4 | 5 | 4.00 |
| D | GEDI + Sentinel-1 regression | 3 | 3 | 3 | 5 | 3.50 |
| E | Photogrammetry/UAV voxels | 2 | 3 | 3 | 2 | 2.50 |

**Read of the table:** A is the safest, highest-validation build; B is the strongest *novelty* with only slightly more risk and the same validation backbone. The two are not mutually exclusive — **A is the core that B extends.** We therefore recommend a staged plan: build A first (guaranteed defensible product), then layer B's SAR occlusion-fill as the headline novelty if time allows.

---

## Recommended MVP (the single best bet)

**Build A as the v0 core, on a single SE-US longleaf-pine site with co-located truth, and extend toward B (SAR occlusion-fill) as the novelty layer.**

### Why this one
- **Maximizes the top-weighted axis (validation).** The RxCADRE 2012 Eglin AFB dataset has the LiDAR **input** (TLS point clouds, `RDS-2023-0011`) and the destructive surface-fuel **truth** (loading/depth/moisture by component, `RDS-2014-0031`) in the *same* longleaf plots — both public-domain, no login ([AgData Commons](https://agdatacommons.nal.usda.gov/articles/dataset/RxCADRE_2012_Terrestrial_laser_scan_TLS_point_cloud_data_for_Eglin_Air_Force_Base/27010462)). This is the cleanest possible apples-to-apples validation.
- **Directly relevant to the sponsor.** Longleaf pine / palmetto-gallberry understory is exactly the SE-US, frequently-burned, DoD-installation fuel type NAWCTSD/SERDP-ESTCP care about.
- **Simple to explain.** LiDAR → 1 m voxel occupancy → linear calibration to clip-plot mass. One paragraph.
- **Buildable now, entirely on open data.** PDAL, Python, free archives.
- **Drop-in FastFuels compatibility.** Outputs the 1 m netCDF bulk-density array (pathway 2) and a layerset GeoJSON (pathway 3).

### v0 specification

**Inputs**
- **Primary (validation site):** RxCADRE 2012 TLS LAZ (`DOI 10.2737/RDS-2023-0011`) for the Highly Instrumented Plots L1G (grass), L2G (grass/shrub), L2F (forested); plus RxCADRE ground-fuel clip/line-intercept measurements (`RDS-2014-0031`).
- **Scalability demonstration site:** NEON Ordway-Swisher (OSBS) airborne discrete-return LiDAR `DP1.30003.001` + CHM `DP3.30015.001`, with field truth `DP1.10023.001` (herbaceous clip), `DP1.10033.001` (litter/fine woody), `DP1.10014.001` (coarse downed wood). One-line download via `neonutilities` (`nu.by_tile_aop(...)`, `nu.load_by_product(...)`).
- **Baseline-to-beat:** LANDFIRE FBFM40 (layer `240FBFM40`) via the LFPS REST API — the modeled 30 m surface layer FastFuels uses.
- **SAR (novelty extension):** Sentinel-1 GRD VV/VH via `asf_search` (free, Earthdata Login); optionally NISAR L+S band.

**Method**
1. Height-normalize each point cloud (ground filter → DTM).
2. Isolate the near-ground band (~0.15–2 m) beneath a canopy mask derived from the CHM.
3. Voxelize at 1 m (and 0.25 m for the surface stratum); compute per-voxel occupied-volume fraction / return density.
4. Fit a **single transparent regression** (linear or small random forest) mapping voxel metrics → surface fuel loading (kg/m²) and depth (m); derive **bulk density = loading / depth**; assign SAVR from species/fuel-type priors; split live/dead from field records.
5. **(B extension)** Stratify by canopy cover; where ALS understory returns vanish, substitute SAR features (VV, VH, VH/VV ratio, temporal mean/CV) with canopy cover as an interaction term.
6. Segment cells into patches; write each as a layerset polygon (`fuel_type`, `fuel_loading`, `fuel_height`, `percent_cover`, `distribution=random_clusters` with `patch_size`/`patch_std_dev` from observed clumping, plus live/dead moisture).

**Outputs**
- A **FastFuels-compatible 1 m voxelized netCDF** bulk-density array (pathway 2), validated by extruding loading/depth into 1 m vertical bins.
- 2D property rasters: loading (kg/m²), depth (m), bulk density (kg/m³), live/dead %, SAVR, percent cover, patch metrics.
- A **layerset GeoJSON** (pathway 3) demonstrated to ingest via `POST /features/layerset/geojson` → `POST /grids/rasterize/layerset` → QUIC-Fire export.

**Validation**
- Spatially **blocked holdout** of plots/sites to prevent leakage.
- Report **R²/RMSE/bias** on loading and depth vs clip plots; target the TLS benchmark bar of **~16% relative RMSE** and ~85% R² of occupied-volume vs destructive mass ([Rowell et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)).
- On the Eglin TLS tile, compute **voxel occupancy IoU** and **vertical-profile correlation** for true 3D agreement.
- **Head-to-head vs FastFuels:** report the coefficient of variation of our 1 m bulk density vs the single uniform value FastFuels assigns the same fuel-model class — the quantitative "heterogeneity FastFuels misses" figure.
- Attach **per-voxel prediction intervals** (quantile RF / ensemble spread) for uncertainty quantification.
- Report metrics **per ecosystem** (Eglin longleaf + OSBS) to evidence transfer.

---

## Build Roadmap (2026-06-23 → 2026-07-20)

~4 weeks. Each week ends with a concrete, demonstrable artifact. Milestones map explicitly to the judging rubric.

> **Progress (2026-06-23):** Weeks 0–2 ✅ ahead of schedule.
> - **W0/W1:** synthetic pipeline + dashboard; **real Eglin 3DEP LiDAR** → measured 1 m grid
>   (CV 0.46) over **real Esri imagery** vs uniform (CV 0). USGS 3DEP via TNM Access is the
>   ungated real-LiDAR on-ramp (RxCADRE hosts were bot-gated).
> - **W2:** **SAR+LiDAR fusion** = `(1−canopy)·LiDAR + canopy·SAR`. Synthetic blocked-CV:
>   **fusion R² 0.71** > LiDAR-only 0.51 > SAR-only 0.37 > uniform 0; under canopy **0.50 vs
>   LiDAR −0.01** (radar fills the occlusion gap). **Real Sentinel-1 RTC** over Eglin (VH/VV +
>   fused product) pulled ungated from Microsoft Planetary Computer.
> - **W3 — the GLOBAL story (two-stage, build order corrected):** a global **30 m product**
>   from spaceborne (Stage 1) is the prerequisite; a mass-conserving **30 m → 1 m downscaler**
>   (Stage 2) sharpens it, both trained on US 3DEP and applied globally. **Downscaler POC done**
>   (regression: within-block R² 0→0.31, mass-conserving) — but it currently uses a coarsened-3DEP
>   *stand-in* for the 30 m input. **Next = build Stage 1 (global 30 m) first**, then re-wire the
>   downscaler to it; later swap the regressor for **AlphaEarth/Clay embeddings + a UNet**.
>   Full architecture + ordered tasks: [research/DOWNSCALING.md](research/DOWNSCALING.md), [TODO.md](TODO.md).

### Week 0 — now → Jun 28: Data, scaffolding, baseline
- Download RxCADRE 2012 TLS (`RDS-2023-0011`) + ground fuel (`RDS-2014-0031`); pull OSBS NEON ALS/CHM/field via `neonutilities`; pull LANDFIRE FBFM40 baseline; register Earthdata Login and pull Sentinel-1 GRD over both sites via `asf_search`.
- Stand up the PDAL/Python pipeline skeleton: ground filter → height-normalize → voxelize.
- **Artifact:** reproducible data-pull scripts + a one-page data inventory with DOIs/URLs and licenses.
- **Rubric:** *data cost* (all free/open), *clarity* (scripted, reproducible).

### Week 1 — Jun 29 → Jul 5: Core voxelizer + calibration (Approach A)
- Voxelize Eglin HIP plots at 1 m and 0.25 m; compute occupied-volume/return-density.
- Fit the loading/depth regression to clip-plot truth; derive bulk density; assign SAVR/live-dead.
- **Artifact:** first 1 m bulk-density netCDF for one plot + calibration scatterplots with R²/RMSE.
- **Rubric:** *validation/credibility* (headline number), *relevance to surface/understory fuels*.

### Week 2 — Jul 6 → Jul 12: Validation harness + FastFuels benchmark + scalability
- Implement spatially blocked holdout; compute R²/RMSE/bias, voxel IoU, vertical-profile correlation, prediction intervals.
- Run the identical pipeline on OSBS ALS (wall-to-wall 1 km) → generality/scalability evidence.
- Overlay LANDFIRE FBFM40 / FastFuels per-class value; compute the CV "heterogeneity missed" figure.
- **Artifact:** validation report (per-ecosystem tables) + side-by-side "uniform vs measured" map.
- **Rubric:** *validation* (top weight), *generality*, *scalability*, *utility*.

### Week 3 — Jul 13 → Jul 18: SAR occlusion-fill (Approach B) + FastFuels ingestion
- Add Sentinel-1 (and NISAR L-band if obtainable) features; train the canopy-cover-interaction fusion model; show R² improvement on closed-canopy plots vs the LiDAR-only ceiling (0.27–0.59).
- Export layerset GeoJSON; demonstrate end-to-end ingestion into FastFuels and QUIC-Fire `.dat` export.
- **Artifact:** fusion-vs-LiDAR-only comparison figure + a working FastFuels layerset round-trip.
- **Rubric:** *novelty*, *utility/FastFuels compatibility*, *relevance*.

### Week 4 — Jul 19 → Jul 20: Dashboard, write-up, submit
- **Dashboard:** an interactive viewer — input LiDAR/SAR, the 1 m bulk-density field, the FastFuels uniform layer, the difference map, and the validation scatterplots/UQ — so judges can see "measured beats modeled" in one screen.
- **Validation artifacts to package:** per-ecosystem R²/RMSE/bias tables; voxel IoU + vertical-profile correlation; heterogeneity-CV comparison; prediction-interval coverage; the head-to-head FastFuels figure.
- Finalize the one-paragraph method statement and the submission package (netCDF + layerset GeoJSON + property rasters + report).
- **Rubric:** *clarity* (dashboard + one-paragraph method), and a final pass confirming every required per-voxel property (loading, bulk density, depth, live/dead %, SAVR, moisture, patch metrics) is present and validated.

### Rubric coverage summary

| Judging criterion (by weight) | How this plan addresses it |
|---|---|
| **Validation / credibility (highest)** | Co-located TLS + destructive truth (RxCADRE); blocked holdout; R²/RMSE/bias + voxel IoU + UQ; head-to-head vs FastFuels uniform value. |
| Utility | FastFuels-compatible 1 m netCDF + layerset GeoJSON; QUIC-Fire `.dat` round-trip. |
| Generality across ecosystems | Eglin longleaf + OSBS; identical re-fit-per-ecosystem pipeline. |
| Data cost | 100% free/open (3DEP, NEON, RxCADRE, Sentinel-1, NISAR, GEDI, LANDFIRE). |
| Clarity | One-paragraph method; scripted; interactive dashboard. |
| Relevance to surface/understory fuels | Entire product targets the occluded under-canopy stratum FastFuels models, not measures. |
| Scalability | Free national/global inputs; wall-to-wall OSBS demo; SAR for repeat coverage. |

### Risks & honest caveats (carry into the write-up)
- **ALS occlusion** under closed canopy caps near-ground accuracy (~R² 0.4) — this is *why* we add SAR; don't hide it.
- **Sentinel-1 is 10 m native**, an order coarser than 1 m — frame SAR as a radar prior *sharpened* by ALS, not a 1 m sensor; do not claim 1 m fuel mass from radar alone.
- **SAR saturates below ~10 Mg/ha** (where surface fuels live) and is moisture-confounded — its most defensible role is occlusion-fill + a live/dead-moisture/curing signal, not standalone retrieval.
- **Temporal offset** between clip-plot dates and LiDAR/SAR acquisition must be documented.
- **Differentiate from ForestGen3D** ([arXiv 2509.16346](https://arxiv.org/abs/2509.16346), the FastFuels/QUIC-Fire team's diffusion-based sub-canopy generator): our under-canopy estimate is anchored to an *independent physical radar measurement* rather than generated from canopy geometry alone. Keep this contrast explicit in related work.
