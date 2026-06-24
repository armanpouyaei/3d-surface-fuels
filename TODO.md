# TODO — build plan

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
- [x] **Real Stage-1 pipeline run** (`build_global30_eglin.py`: AEF+S1+terrain → 3DEP 30 m fuel
      proxy, blocked CV). ⚠️ **Weak (R² ~0.14, AEF-only negative)** — 2024 AEF vs **2007** LiDAR
      (17-yr temporal gap) + tiny homogeneous tile. Pipeline works; the *data vintage* is the issue.
- [ ] **▶️ NEXT: vintage-matched, larger, field-validated run — NEON OSBS** (recent AOP LiDAR
      ~2021-2023 to match AEF year + FIA/NEON field truth + longleaf). This is what makes the real
      result credible.
- [ ] **⚠️ DO FIRST — define + pre-validate the 3DEP→surface-fuel-load equation** against FIA/NEON
      (1 m fuel proxy: fuelbed depth + sub-canopy return density → kg/m²). *Everything downstream
      depends on this; if it's weak (~R² 0.3–0.4) re-scope the variable.*
- [ ] **Scope to ONE 3DEP-rich ecoregion** for the POC; make 3DEP point-cloud processing the Week-1 deliverable.
- [ ] **Ungated predictors first** (no auth): Sentinel-1 seasonal + Sentinel-2 + GLO-30 terrain via Planetary Computer → real 30 m predictor stack over the region.
- [ ] **Auth-gated upgrades** (provide creds): **AlphaEarth** Satellite Embedding (GEE: `earthengine` auth) — the key feature; **GEDI** L2A/L2B PAVD + L4A (NASA Earthdata via `earthaccess`).
- [ ] Train `global30` on the real stack; spatially-blocked + leave-one-ecoregion-out CV; **per-stratum** metrics; held-out FIA/NEON (quote usable n); AEF ablation **with leakage caveat**.
- [ ] **Output**: 30 m fuel median+quantiles per AOI → `data/processed/global30_*.tif`; script `scripts/build_global30.py`.
- [ ] *(Deferred from POC per critique: Clay self-hosting; from-scratch EfficientNetV2 baseline — keep as later axes.)*

## Stage 2: Re-wire the downscaler to the real 30 m product
- [ ] Feed the **Stage-1 30 m product** (not coarsened ALS) into `downscale.downscale_cv` as the coarse baseline
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
