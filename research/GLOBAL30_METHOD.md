# Stage 1: 30 m Surface-Fuel Product — method

> A continuous **surface/understory fuel-load** product (kg/m²) at **30 m** from free
> spaceborne inputs, trained on US airborne-LiDAR (3DEP) targets + field plots (FIA DWM,
> NEON). **Scope: region/AOI prediction is the required capability; global is an optional
> production scale-up the design supports (tiled per-AOI), not a validated global number.**
> Feeds the Stage-2 30 m→1 m downscaler. Reconciled with an adversarial critique — see
> "Honest framing" before quoting any number.

## The 2025 reference (and why it's not a like-for-like benchmark)

**de Conto, Armston & Dubayah (2025)** — *"Scalable deep fusion of spaceborne lidar and SAR
for global forest structural complexity mapping"* ([arXiv:2510.06299](https://arxiv.org/abs/2510.06299);
*Machine Learning: Earth* 2(1):015002).

**Method.** Fully-convolutional **EfficientNetV2** (**365,682 params**), input **40×40×10 @ 25 m**
(ALOS PALSAR L-band HH/HV + Sentinel-1 C-band VV/VH + Copernicus DEM + 3 coordinate channels),
output 32×32×2 (mean+variance). Trained on **~5 M 1 km chips / ~133 M GEDI footprints** (2019–2022),
**80/20 spatially-blocked (80 km) split**, **masked Gaussian NLL loss**, MC-dropout for epistemic
uncertainty (~200 GPU-h). Global GEDI-vs-GEDI **R²=0.82**; calibration 71%/96% at 1σ/2σ.

**The crucial finding: it does not predict fuel.** The target is the GEDI L4C **Waveform
Structural Complexity Index (WSCI)** — a unitless canopy index ([de Conto 2024, *Nat. Commun.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405527/)).
The word "fuel" appears **zero** times. So we **cannot "beat" R²=0.82** — it's a different
variable. We adopt their *engineering* and **retarget to surface fuel**.

**Openings we exploit** (their stated gaps): no optical/Sentinel-2; no field plots; no InSAR;
**no GEDI L2B/L4A** (no PAVD/understory metrics); saturates below WSCI 7.4 (the low-stature
regime where surface fuels live); GEDI product uncertainties not propagated; **0.82 is
circular** (GEDI-predicting-GEDI) — against independent ALS it is **R²≈0.47 (0.56 fine-tuned)**.

**The real fuel-domain bar.** [Leite et al. 2022, *RSE*](https://www.sciencedirect.com/science/article/abs/pii/S0034425721004843)
(GEDI, Cerrado, 1 km RF): woody R²=0.88, total 0.71, but **surface fuels only R²≈0.31**.
[Labenski et al. 2023](https://research.fs.usda.gov/treesearch/66562): litter R²=0.27–0.41,
herb/shrub 0.55–0.64. **No published continuous 30 m surface-fuel-load product exists — that's the white space.**

## Honest framing (apply throughout)

- We build **the first continuous 30 m surface-fuel-load product**; we **benchmark our method
  against de Conto's *architecture trained on our fuel target*** (same folds) — not against 0.82.
- **Biggest risk:** the **3DEP → surface-fuel-load equation is a proxy of a proxy** (airborne
  LiDAR under-senses litter/duff/fine-woody under canopy). **It must be defined and
  pre-validated against FIA/NEON in Week 1.** If it's weak (~R² 0.3–0.4, Labenski's ceiling),
  the whole "surface fuel" framing is downgraded. *Everything depends on this.*
- **AlphaEarth leakage:** AEF already ingests GEDI/S1/PALSAR/DEM and is pretrained on the test
  regions, and it's spatially continuous — so spatial-block CV does **not** fully prevent
  embedding leakage, and the "physical-only vs +AEF" ablation cannot cleanly attribute gains
  (it may re-package GEDI). State this; treat AEF gains as **to-be-measured**, not assumed.
- **Field validation is thin:** FIA coords are fuzzed ~1 km (areal anchor, not pixel-level);
  NEON is precise but ~tens of sites. Quote the expected usable held-out **n**, don't imply
  wall-to-wall field truth.
- **Global accuracy is unvalidated** (no global ground-fuel truth). The product is
  **region/CONUS-validated, globally-extrapolated with an OOD flag** — not "global 30 m fuel."

## Our method

**Target:** surface/understory **fuel load, kg/m²**, at **30 m** (EPSG:5070 for CONUS;
arbitrary AOI/UTM for region runs), predicted as **median + 0.05/0.95 quantiles**. Optionally
split litter/duff · fine/coarse down-woody · herb · shrub as multi-output heads — **report
per stratum, never blended into a single inflated total.**

**Predictors** (free; harmonized to the 30 m AOI grid):
- **AlphaEarth Foundations / Satellite Embedding** — 64-band, 10 m, annual (2017–2025),
  unit-length embeddings. **Free, ungated, no Earth Engine, no payment (incl. commercial; CC-BY-4.0,
  attribute Google/DeepMind)** via the public **Source Cooperative** mirror `tge-labs/aef`
  (verified readable). Two access paths:
  (a) per-UTM-zone COGs `…/tge-labs/aef/v1/annual/{year}/{zone}/{img}-{yoff}-{xoff}.tiff` — **read
  the sibling `.vrt`, not the raw `.tiff`** (COGs are stored *bottom-up*; the `.vrt` corrects the
  flip; geolocation is in the COG/VRT georef + the opaque image-hash name, not the offset);
  (b) global **Zarr mosaic** `tge-labs/aef-mosaic` (EPSG:4326, xarray-openable) — simplest AOI query.
  **De-quantize** int8→float: `((v/127.5)**2)*sign(v)`; NoData=−128. Primary feature block,
  mean-pooled 10 m→30 m; use a **nonlinear/conv head, not a linear probe.** (GEE asset
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` exists too but isn't needed — avoids its commercial cost.)
- **Explicit physical predictors** (kept for SHAP/explainability + ablation, and as the
  no-AEF fallback): GEDI **L2A RH** + **L2B PAI/FHD/cover and PAVD 0–5 m & 5–10 m bins**
  (the understory/ladder signal the SOTA omits; *needs Earthdata*); GEDI **L4A AGBD** prior;
  **Sentinel-1 seasonal** VV/VH (dry-season min + amplitude → curing; Planetary Computer,
  ungated); **Sentinel-2** NDVI/NDII/NBR phenology (PC, ungated); **GLO-30** slope/aspect/TWI
  (PC, ungated); RESOLVE ecoregion one-hot (biome conditioning).
- **Clay v1.5** embeddings — *deferred past the POC* (self-hosting cost); future axis.

**Model.** POC: **LightGBM/HistGBM with quantile (pinball) objective** — fast, explainable
(SHAP on named features), tabular over the 30 m cells. Upgrade: small **conv/UNet head over
frozen embeddings** for spatial context. **Softplus/clip → non-negativity.**

**Loss / uncertainty.** Masked **Gaussian NLL + pinball** (handles GEDI sparsity, gives
quantiles); **split-conformal calibration grouped by ecoregion** (a coverage guarantee **only
on the region's own distribution** — not a global guarantee); **embedding cosine-distance to
nearest training pixel = OOD flag** for out-of-region application.

### de Conto's approach → ours (each gain is *measured*, not asserted)

| Axis | de Conto 2025 | Ours | Status |
|---|---|---|---|
| Target | WSCI (unitless index) | **fuel load kg/m²** (per stratum) | different variable; fills white space |
| Features | raw PALSAR+S1+DEM, from scratch | **AEF embeddings** + explicit GEDI L2B/L4A + seasonal S1/S2 | measure Δ via ablation (AEF-leakage caveat) |
| Understory | RH only; saturates <7.4 | **PAVD 0–5/5–10 m, L-band, optical phenology** | targets the hard stratum |
| Labels | GEDI self-supervision | **3DEP→30 m + FIA/NEON** field anchor | pending 3DEP→fuel pre-validation |
| Uncertainty | NLL + MC-dropout | NLL + quantiles + **ecoregion conformal + OOD flag** | region coverage; not global guarantee |
| Reporting | headline R²=0.82 (circular) | held-out blocks + ecoregions + field plots, **per stratum** | honest, lower, defensible |

## Data & training recipe

**Targets:** USGS **3DEP** (Entwine `s3://usgs-lidar-public` / PC `3dep-lidar`, ungated) → 1 m
fuel proxy → 30 m kg/m²; **FIA DWM** (`rFIA::dwm()`, areal anchor); **NEON** CWD/litter/herb
(`neonUtilities`, precise validation). **Predictors:** AEF (GEE), GEDI L2A/L2B/L4A
(`earthaccess`), S1-RTC + S2-L2A + GLO-30 + WorldCover/ecoregion (Planetary Computer, ungated).

**Auth needed for the full method:** **Google Earth Engine** (AlphaEarth) and **NASA Earthdata**
(GEDI). Everything else (3DEP, Sentinel-1/2, GLO-30 via Planetary Computer) is ungated — so an
**ungated POC** (S1 seasonal + S2 + terrain → 3DEP 30 m fuel) is runnable now without auth, and
AEF/GEDI slot in once credentials are provided.

## Validation plan
1. **Spatially-blocked CV** (GroupKFold by ~80 km blocks) — kills label autocorrelation.
2. **Leave-one-ecoregion-out** (RESOLVE biome) — transfer proxy (still within-CONUS; not global).
3. **Per-biome AND per-stratum** metrics (litter/duff vs woody vs herb/shrub). Bars to beat:
   Leite surface R²≈0.31; Labenski litter 0.27–0.41 / herb-shrub 0.55–0.64.
4. **Held-out FIA/NEON** field plots — quote expected usable **n**; the credibility differentiator.
5. **Calibrated uncertainty** — Z-coverage (≈69/95%), ecoregion conformal coverage, σ-vs-|resid|, OOD-vs-error.
6. **Head-to-head:** de Conto's architecture **trained on our fuel target, same folds** (budget honestly — a rushed from-scratch CNN is a strawman; may defer). **Ablation:** physical-only vs +AEF (with the leakage-attribution caveat). **vs LANDFIRE FCCS** as a categorical context baseline.

## 4-week POC plan (1 ecoregion → region capability; global = tiled extrapolation)
- **W1 — Target first.** Process 3DEP over **one** 3DEP-rich ecoregion; **define + pre-validate
  the 3DEP→surface-fuel-load equation against FIA/NEON.** Gate the rest of the plan on this.
- **W2 — Predictors + explainable model.** Assemble the 30 m predictor stack (start ungated:
  S1 seasonal + S2 + terrain; add AEF/GEDI if auth available). Train **LightGBM-quantile**;
  spatially-blocked CV; SHAP.
- **W3 — Uncertainty + ablations.** NLL/pinball, deep ensemble, ecoregion conformal, OOD flag;
  AEF ablation; per-biome/per-stratum tables; validate vs held-out FIA/NEON.
- **W4 — Honest report + region/AOI interface.** Leave-one-ecoregion-out; SOTA-architecture-on-
  fuel head-to-head; clean **30 m mean+quantile AOI output** for Stage 2; (optional) a couple of
  out-of-region tiles with OOD flags as the *capability* (not accuracy) demo.

**Stay simple/explainable:** frozen embeddings (no foundation-model training), tree-based or
<1 M-param heads, SHAP on named physical features, soft (not hard) priors.

## Honest risks
- **3DEP "fuel" is a proxy** that under-senses sub-canopy fuel → bounds true accuracy; pre-validate vs FIA/NEON.
- **AEF gains may be modest** for surface fuel (sub-canopy; AEF's LiDAR input is GEDI) and partly **leaked**; the ablation measures this — if small, fall back to the explicit-feature story.
- **Field validation is small-n** (FIA fuzzed, NEON sparse); state n.
- **Global accuracy unvalidated** — region-validated, globally-extrapolated with OOD flags.
- **Honest CV reads far below 0.82** — that's the point: a credible, defensible number.

See [DOWNSCALING.md](DOWNSCALING.md) for Stage 2, [../TODO.md](../TODO.md) for tasks.
