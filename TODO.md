# TODO — build plan

## ✅ Deliverables audit vs official challenge page (2026-06-26)

Checked our repo against the required deliverables on
https://centralfloridatechgrove.org/surface-fuels-prize-challenge/ (Phase-1 deadline 2026-07-20):

| deliverable | status |
|---|---|
| 1 m³ geo-referenced arrays / FastFuels-compatible NetCDF (sentinel 1.23456) | ✅ `build_deliverable_osbs.py` |
| P1 fuel load (kg/m²) — field-calibrated | ✅ |
| P1 cell bulk density (kg/m³) | ✅ |
| P1 **percent cover (%)** exported | ⚠️ computed, not exported → **T18** |
| P2 SAVR · live/dead (%) | ✅ SB40 lookup |
| P2 **dead + live fuel moisture (%)** | ❌ sentinel → **T21 (ERA5/CDS)** |
| P3 patch size · heat of combustion · <2 m heterogeneity · species | ⚠️/❌ → **T22** |
| **Boundary polygon GeoJSON** (+ projection) | ❌ → **T18** |
| Fuel property maps + 3D viz | ✅ dashboard + figures |
| **Validation documentation** (top-weighted) | ✅ RESULTS.md/VALIDATION.md (strong) |
| **Python ingestion/visualization tool** (standalone + README) | ⚠️ pieces exist → **T19** |
| **Consolidated Phase-1 submission package/report** | ❌ → **T20** |

Gaps → tasks: **T18** boundary GeoJSON + % cover · **T19** package the Python tool · **T20**
assemble the submission package/report · **T21** fuel moisture via ERA5/CDS · **T22** cheap P3 props.
Strong on the highest-weighted axes (validation, clarity, generality, low-cost data); remaining work
is required *artifacts/packaging* + a few missing properties.

## 🎯 Prioritized roadmap to submission — deadline **2026-07-20** (~4 weeks)

Goal: **win.** Judged on utility, generality, data cost, clarity, **validation
credibility**, relevance to surface/understory, scalability. The method + dashboard are
built and validated (see [research/RESULTS.md](research/RESULTS.md)); the gaps now are
**field-calibrated credibility** and the **required submission artifacts**.

### P0 — submission-critical (must land before Jul 20)
- [ ] **P0.1 — Validation document** (the most differentiating artifact). Turn RESULTS.md
      into a formal write-up: truth datasets, spatially-blocked-CV methodology, metrics
      (R²/RMSE, within-block, CV/heterogeneity, conformal coverage), per-stratum, the
      FastFuels + de Conto head-to-heads, generality (OSBS↔SOAP), and an explicit
      limitations/uncertainty section. Lead with it.
- [ ] **P0.2 — Field-calibrate structure → absolute load (kg/m²).** Today magnitudes are
      literature-anchored. Calibrate + report **real R²/RMSE vs destructive/TLS/field**
      truth — RxCADRE clip plots (`RDS-2014-0031`), NEON herb (`DP1.10023`), SERDP
      RC19-1064 TLS. Removes the biggest credibility gap. *(Honest: proxy = woody/under-
      story structure; calibrate per stratum, don't conflate with herb.)*
- [ ] **P0.3 — Python visualization/ingestion tool** (required deliverable). Package the
      dashboard + a clean, documented standalone API: read/write FastFuels **Option C**
      NetCDF, load any of our products, render 3D + maps. Pip-installable / one-command run.
- [ ] **P0.4 — Finalize + validate FastFuels exports.** Confirm our 1 m NetCDF ingests via
      the FastFuels API (Option C) and emit a **layerset GeoJSON (Option D)**. Re-run the
      **live FastFuels surface head-to-head** once their LANDFIRE backend recovers
      (`fastfuels_headtohead.py` is API-ready).
- [ ] **P0.5 — Submission package**: georeferenced 1 m arrays + property maps + 3D
      renderings + the validation doc + the viz tool, assembled per the challenge formats.

### P1 — strengthen the win (generality + completeness)
- [ ] **P1.1 — Add 3–5 more 3DEP+AEF training sites** across ecosystems (eastern hardwood
      e.g. HARV, western shrub/grassland, more southern pine). Widens in-distribution,
      improves the portable model, hardens the **generality** claim; re-report leave-one-
      site-out (currently weak: −0.18/−0.45 with only 2 opposite sites).
- [ ] **P1.2 — Total surface load**, not just structure: add **litter + fine/coarse woody
      debris** components; reconcile herb vs woody; report total kg/m² (the challenge's
      headline quantity) with per-component breakdown.
- [ ] **P1.3 — Real SAR+LiDAR fusion end-to-end** (currently synthetic fusion + real S1
      separate). Demonstrate the under-canopy fill on real data, with metrics.
- [ ] **P1.4 — de Conto head-to-head at training scale** (train their CNN on pooled multi-
      site data) so the comparison isn't only AOI-scale where the CNN is data-starved.

### P2 — stretch / post-submission
- [ ] **P2.1 — Stage-2 UNet/Clay downscaler** (replace per-pixel model-reuse with a learned
      UNet on Clay/AlphaEarth embeddings + mass-conservation loss; `torch`+MPS ready).
- [ ] **P2.2 — GEDI regional understory PAVD** as a covariate (sparse → regional interpolation).
- [ ] **P2.3 — Global production scale-up** (tiling / cloud-native) — design-supported;
      demonstrate on one tiled region.
- [ ] **P2.4 — Conformal-coverage + OOD validation report** across held-out sites.

---

## Build-log / detailed status (below is the chronological record)

Ordered. **Stage 1 (global 30 m product) before Stage 2 (downscaler).** See
[research/DOWNSCALING.md](research/DOWNSCALING.md) for the architecture.

## ✅ Done
- [x] Research + strategy docs (RESEARCH / VALIDATION / IDEAS / DOWNSCALING)
- [x] `surface_fuels` package: 1 m voxel model, metrics, NetCDF I/O
- [x] Real USGS 3DEP LiDAR over Eglin → measured 1 m grid + real Esri basemap
- [x] SAR + LiDAR fusion (canopy-weighted), blocked-CV (fusion R² 0.71)
- [x] Real Sentinel-1 RTC over Eglin (Planetary Computer)
- [x] **Downscaler POC (regression)** — `downscale.py`, synthetic within-block R² 0→0.31, mass-conserving
  - ⚠️ uses a *coarsened-3DEP stand-in* for the 30 m input → replace with the real Stage-1 product

## Stage 1: 30 m surface-fuel product

> **Scope:** required capability = **small-region (AOI) prediction**; global = optional
> production scale-up the design supports (tiled per-AOI), *not* a validated global number.
> **Framing (per critique):** this is the **first continuous 30 m surface-fuel product**; we
> benchmark our method against **de Conto's architecture trained on our fuel target** — NOT
> "beat their 0.82" (they predict a structure index, not fuel). Honest bars: surface-fuel
> R²≈0.31 (Leite 2022), litter 0.27–0.41 (Labenski). Method: [research/GLOBAL30_METHOD.md](research/GLOBAL30_METHOD.md).

- [x] **Method designed** (de Conto deep-dive + adversarial critique) → `research/GLOBAL30_METHOD.md`
- [x] **Stage-1 harness POC** — `global30.py`: quantile GBM, spatially-blocked CV, conformal
      intervals, OOD flag, per-stratum metrics. Synthetic: embed-only R² 0.42 > physical-only
      0.27, full 0.56; under-canopy honestly lower; `scripts/run_global30_demo.py`.
- [x] **AlphaEarth fetcher — FREE, no GEE** (`embeddings.py`): Source Coop mirror, .vrt-indexed
      COGs, dequantized → (64,ny,nx) for any AOI grid. Verified over Eglin.
- [x] **Real Stage-1 pipeline run** (`build_global30.py`: AEF+S1+terrain → 3DEP 30 m fuel
      proxy, blocked CV). ⚠️ **Weak (R² ~0.14, AEF-only negative)** — 2024 AEF vs **2007** LiDAR
      (17-yr temporal gap) + tiny homogeneous tile. Pipeline works; the *data vintage* is the issue.
- [x] **Vintage-matched real run — OSBS, 2018 3DEP + AlphaEarth 2018** (`build_global30.py --site osbs`):
      **R² 0.14 → 0.61** (confirms the temporal gap was the issue), interval coverage 0.90. **Honest
      ablation finding: AlphaEarth adds ~nothing over Sentinel-1 + terrain** (full 0.607 vs physical
      0.604; small under-canopy bump 0.06 vs −0.05; AEF-only 0.15). AEF overlaps S1 and its structure
      signal is GEDI/canopy-top → limited understory info. Don't over-rely on AEF; the ablation is the
      credibility tool.
- [x] **AEF orientation bug fixed** — `aef_for_grid` returned north-up vs row 0 = south
      target/S1 → AEF was fed flipped. Fixed; **REVERSES the earlier "AEF adds nothing":**
      OSBS AEF-only R² **0.15 → 0.66** (beats physical 0.60), full **0.72**, AEF rescues
      under-canopy (0.28 vs −0.05). `figures/global30_osbs.png`.
- [x] **#1 NEON field validation** (`build_neon_field.py`): 79 OSBS herb clips (2018-19),
      leave-one-plot-out — **AlphaEarth predicts field herb load R² 0.23 / Spearman 0.51**,
      beats S1 (0.07). Validated on *real ground fuel*, not a proxy. `figures/neon_field_validation.png`.
      (Caveat: herb component only, n≈20 plots, 0.2 m² clip vs 10 m pixel.)
- [x] **#2 GEDI L2B PAVD** (`build_gedi_osbs.py`, token works): extracted real understory PAVD
      (0-5 m / 5-10 m) + PAI/cover over OSBS. **Finding: too sparse for a dense 30 m predictor** —
      ~395 footprints/pass over the ~11 km box but only **3 in the 1 km AOI**. → use GEDI at
      REGIONAL scale (sparse training labels / interpolated covariate over many passes), not per-tile.
      The understory PAVD signal AEF/S1 lack is real; `figures/gedi_osbs.png`.
- [x] **Proxy validation (CRITICAL)** — `validate_proxy_field.py`: the 3DEP near-ground proxy
      does **NOT** track field herb load (Pearson 0.13 ns, Spearman 0.02). It measures woody/shrub
      understory, not herbaceous. So Stage-1's R² 0.66/0.72 predicts the *proxy*, not validated herb
      fuel. Predictors are fine (AEF→field herb Spearman 0.51); the **proxy target is the problem**.
      `figures/proxy_vs_field.png`.
- [ ] **▶️ PIVOT (pending user nudge): field-calibrated targets, not the 3DEP proxy.**
      (a) **Multi-ecosystem generality** = AEF → NEON field herb across ~3-5 NEON sites (all 81 have
      herb clips) — the validated path + the generality axis; (b) treat the 3DEP near-ground proxy as
      a separate *woody-understory* component (legit, just not herb); (c) Stage-2 rewire; (d) de-Conto
      head-to-head. NEON litter (DP1.10033) is a FLUX, not standing load — can't naively sum for "total".
- [ ] **⚠️ DO FIRST — define + pre-validate the 3DEP→surface-fuel-load equation** against FIA/NEON
      (1 m fuel proxy: fuelbed depth + sub-canopy return density → kg/m²). *Everything downstream
      depends on this; if it's weak (~R² 0.3–0.4) re-scope the variable.*
- [ ] **Scope to ONE 3DEP-rich ecoregion** for the POC; make 3DEP point-cloud processing the Week-1 deliverable.
- [ ] **Ungated predictors first** (no auth): Sentinel-1 seasonal + Sentinel-2 + GLO-30 terrain via Planetary Computer → real 30 m predictor stack over the region.
- [ ] **Auth-gated upgrades** (provide creds): **AlphaEarth** Satellite Embedding (GEE: `earthengine` auth) — the key feature; **GEDI** L2A/L2B PAVD + L4A (NASA Earthdata via `earthaccess`).
- [ ] Train `global30` on the real stack; spatially-blocked + leave-one-ecoregion-out CV; **per-stratum** metrics; held-out FIA/NEON (quote usable n); AEF ablation **with leakage caveat**.
- [ ] **Output**: 30 m fuel median+quantiles per AOI → `data/processed/global30_*.tif`; script `scripts/build_global30.py`.
- [ ] *(Deferred from POC per critique: Clay self-hosting; from-scratch EfficientNetV2 baseline — keep as later axes.)*

## HEADLINE (user steer 2026-06-24): MEASURED 3D STRUCTURE / BULK-DENSITY
> Multi-ecosystem test confirmed herbaceous *load* from spaceborne is genuinely hard
> (transfer ρ 0.25, unstable — literature ceiling). Fuel *structure* is robust. So the
> deliverable is the **measured 3D fuel-structure/bulk-density** product that beats
> FastFuels' uniform surface layer; herb load = honest uncertainty-flagged add-on.

- [x] **END-TO-END pipeline over OSBS** (`build_pipeline_osbs.py`) — Stage-1 (spaceborne
      AEF+S1+terrain → 30 m structure, blocked CV **R² 0.767**, open 0.79 / under-canopy 0.48)
      → Stage-2 (downscale the *predicted* 30 m → 10 m with AlphaEarth 10 m + terrain) →
      validated vs **measured 3DEP** truth, head-to-head vs FastFuels uniform.
      RESULT: end-to-end **R² 0.664, within-block R² 0.274** (the sub-30 m detail FastFuels
      structurally lacks — its within-block R² = 0); heterogeneity CV 0.73 vs truth 0.89;
      FastFuels uniform R² ≈ 0, CV 0. Clean-coarse downscaler 0.852/0.291 = headroom if the
      30 m baseline improves. `figures/pipeline_osbs.png`, `data/processed/pipeline_osbs.npz`.

- [x] **de Conto (2025) architecture head-to-head** (`deconto_headtohead.py`) — faithful
      compact reproduction of their fully-conv EfficientNetV2 (Fused-MBConv→MBConv+SE,
      Gaussian-NLL, MC-dropout, 64,242 params) vs our per-pixel quantile GBM, on the **same**
      fuel-structure target / 68-ch stack / 2×2 blocked folds. **Ours R² 0.628 (cov 85%) vs
      CNN −0.147 (cov 14%)** — at AOI scale the per-pixel model wins; the CNN is data-starved
      (their edge needs continental training, which our tiled design supports). We also add
      calibrated UQ + per-stratum + OOD their WSCI product lacks. `figures/deconto_headtohead.png`.
      NOTE: CNN runs on **Apple GPU (MPS)** — CPU torch here has no MKL-DNN (~50× slower; 1 s/step).

## Stage 2: Re-wire the downscaler to the real 30 m product
- [x] Feed the **Stage-1 30 m product** (not coarsened ALS) into the downscaler as the coarse
      baseline — done in `build_pipeline_osbs.py` (`end_to_end_downscale`: trains on true coarse,
      deploys on predicted coarse, mass-conserves to the deployed 30 m). Eglin POC superseded.
- [ ] Add globally-available 10 m covariates: **AlphaEarth Foundations** (GEE Satellite Embedding) and/or **Clay** ViT embeddings, + Sentinel-2, terrain
- [ ] Re-validate over Eglin + ≥1 other ecosystem (held-out); report within-block R², CV, mass-conservation
- [ ] Dashboard: add **"30 m → 1 m downscaler"** mode (Truth | Stage-1 30 m | Downscaled, synced; metrics) — *not yet built*

## Stage 2b: Upgrade model — Clay/AlphaEarth + UNet
- [ ] Swap the per-pixel regressor for a **UNet** super-resolver (keep the `downscale.py` feature-stack + mass-conservation + blocked-CV interface)
- [ ] Inputs: AlphaEarth/Clay embedding channels + Stage-1 coarse conditioning channel
- [ ] **Mass-conservation loss** so the 1 m output disaggregates exactly to the 30 m input
- [ ] Train on 3DEP 1 m tiles (multi-ecosystem); validate on held-out ecosystems; deps: `torch`, `claymodel`, GEE/`earthengine-api`

## Cross-cutting (anytime)
- [ ] **RxCADRE clip plots** (manual download — hosts are bot-gated) for absolute kg/m² calibration + real R²/RMSE
- [ ] **Cross-ecosystem transfer test** (run the pipeline on 2–3 NEON sites; per-ecosystem metrics) — hardens the "generality" claim
- [ ] NEON OSBS recent co-registered LiDAR + Sentinel-1 (removes the 2007-vs-2026 temporal gap)
- [ ] FastFuels API head-to-head; export FastFuels Option C (1 m NetCDF) / Option D (layerset GeoJSON)
- [ ] Submission package: 3D fuel arrays, property maps, validation doc, Python viz tool (deadline 2026-07-20)

## Notes / gotchas (so you don't re-derive)
- Ungated data: USGS 3DEP via **TNM Access API**; Sentinel-1 RTC via **Planetary Computer** (`planetary_computer.sign_inplace`); Esri imagery export. Gated/avoid: AgData Commons & FS RDS (RxCADRE — bot-gated), ASF, OpenTopography (key).
- Dashboard map panels: `go.Image` forces shared y-axis reversed → all arrays `np.flipud`-ed to row 0 = north; axes `matches` for synchronized zoom.
- Eglin 3DEP tile is 2007 (low density); Sentinel-1 is 2026 → temporal gap is real; the *synthetic* experiments are the clean method validations.
