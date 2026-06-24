# Global Surface-Fuel Product: Two-Stage Architecture

> How we make the approach **global**, not local. Build order matters:
> **Stage 1 (global 30 m product) comes first**, then **Stage 2 (30 m → 1 m downscaler)**
> consumes it. The downscaler is meaningless without a real coarse input — coarsening a
> 3DEP grid is only a POC stand-in.

```
                    ┌─────────────────────────── STAGE 1 (build FIRST) ───────────────────────────┐
  GEDI L2A/L2B/L4   │                                                                              │
  ICESat-2 ATL08    │   spaceborne predictors ──► [model] ──► GLOBAL 30 m surface-fuel product     │
  Sentinel-1/NISAR  │   (global, free)                         (kg/m², + per-cell uncertainty)     │
  Sentinel-2        │                              ▲                                               │
  terrain / biome   │              target = 3DEP 1 m fuel aggregated to 30 m (US)                  │
                    └──────────────────────────────────────────────┬───────────────────────────────┘
                                                                    │  (coarse baseline, available everywhere)
                    ┌──────────────────────────── STAGE 2 ──────────▼───────────────────────────────┐
  Sentinel-1/2 10 m │   coarse 30 m fuel + 10 m covariates ──► [downscaler] ──► 1 m surface fuel     │
  AlphaEarth/Clay   │   (+ AlphaEarth/Clay embeddings)          mass-conserving  (+ uncertainty)     │
  terrain, canopy   │                              ▲                                                 │
                    │              target = 3DEP 1 m fuel (US)                                       │
                    └────────────────────────────────────────────────────────────────────────────────┘
```

**Why this order.** The whole point is a product that exists *everywhere*. That requires a
coarse fuel estimate from globally-available spaceborne data (Stage 1). The downscaler
(Stage 2) sharpens that estimate to 1 m using globally-available 10 m covariates. Both
stages are **trained over the US** (where 3DEP gives wall-to-wall targets at both 30 m and
1 m) and **applied globally** (because every *input* is a global sensor — only the *target*
is US-limited).

---

## Stage 1 — Global 30 m surface-fuel product *(build first)*

Produce a wall-to-wall 30 m estimate of surface/understory fuel load (kg/m²) from
spaceborne sensors, with per-cell uncertainty.

| | |
|---|---|
| **Predictors (global, free)** | GEDI L2A relative-height + L2B PAI/PAVD vertical profiles; ICESat-2 ATL08 (high latitudes); Sentinel-1/NISAR backscatter (VV/VH temporal mean+CV, ratio); Sentinel-2 seasonal indices; terrain (Copernicus DEM); biome / LANDFIRE EVT as a categorical |
| **Target (US, for training)** | 3DEP-derived 1 m fuel **aggregated to 30 m** (wall-to-wall over CONUS); densified with field plots (FIA Down Woody Material, NEON herb/litter/CWD, RxCADRE) for absolute calibration |
| **Model** | start simple/explainable — gradient-boosted / quantile RF regression (consistent with the POC); later a foundation-model regressor. Condition on biome to aid transfer |
| **Validation** | spatially-blocked + **held-out ecosystem** CV; report R²/RMSE per biome; calibrated uncertainty (≈90% interval coverage); flag out-of-distribution inputs |
| **Output** | global 30 m fuel-load raster + uncertainty — the coarse baseline that feeds Stage 2 |

**Honest expectation:** spaceborne surface-fuel skill is modest (GEDI alone ≈ R² 0.15–0.46
on the surface stratum); the 30 m product is a *prior*, not a measurement. Multi-sensor
fusion (GEDI+SAR) reaches R² ≈ 0.82 for *structure* (de Conto 2025) — surface fuel will be
lower, which is exactly why Stage 2 + ALS calibration matter.

---

## Stage 2 — 30 m → 1 m downscaler *(POC done; re-wire to Stage 1)*

Sharpen the Stage-1 30 m product to 1 m using globally-available 10 m covariates, trained
on 3DEP 1 m targets. Implemented in [`src/surface_fuels/downscale.py`](../src/surface_fuels/downscale.py).

- **Mass-conserving:** the 1 m field is rescaled so each 30 m block averages back to the
  Stage-1 value (true disaggregation, not invention).
- **What it can/can't do:** recovers the *recoverable* sub-30 m band (10–30 m structure from
  10 m covariates); the finest <10 m detail is information-limited globally — so the
  ALS-backed 1 m product stays the gold standard where airborne LiDAR exists.

**POC status (regression, validated):** synthetic blocked-CV — overall **R² 0.58 → 0.71**,
within-block (sub-30 m) **R² 0 → 0.31**, mass-conserving (agg R² 1.0). The 10 m
embedding/optical covariate dominates feature importance (~0.8) — exactly where AlphaEarth/
Clay plug in. *Caveat:* the POC's coarse input is a **coarsened 3DEP grid (stand-in)**;
replace it with the real Stage-1 product.

### Future model (planned): Clay/AlphaEarth embeddings + UNet
Swap the per-pixel regressor for a spatial super-resolution network (the
`downscale.py` interface — feature stack + mass-conservation + blocked-CV — stays fixed):
- **AlphaEarth Foundations** embeddings (Google DeepMind; annual global **10 m, 64-D**
  "Satellite Embedding" on GEE) and/or **Clay** v1 ViT embeddings as the input covariate
  channels — they encode multi-sensor land-surface state globally, far richer than raw bands.
- **UNet** (encoder–decoder, skip connections) for 30 m → 1 m super-resolution, with the
  Stage-1 coarse field as a conditioning channel and a **mass-conservation loss** so the
  output disaggregates exactly. Train on 3DEP 1 m tiles; validate on held-out ecosystems.
- Differentiator vs **ForestGen3D** (diffusion sub-canopy from ALS, US-input-only): we
  downscale a *global* coarse product and use ALS only as the *training target* — strictly
  more general.

---

## Data availability (global vs US)

| Layer | Stage | Global? |
|---|---|---|
| GEDI, ICESat-2, Sentinel-1/2, NISAR | 1 predictors | **global, free** |
| AlphaEarth / Clay embeddings (10 m) | 2 covariates | **global, free** |
| Copernicus DEM (terrain) | 1+2 | **global, free** |
| 3DEP airborne LiDAR (1 m target) | 1+2 training | **US only** (+ UK/NL/DK/ES/AU national programs) |
| Field plots (FIA/NEON/RxCADRE) | calibration | sparse; US-strong |

## Open risks
1. **Covariate shift** to ecosystems absent from US training (e.g., miombo, eucalypt) — mitigate with biome conditioning, UQ, and folding in non-US ALS.
2. **Stage-1 surface-fuel skill** is inherently modest from spaceborne — set expectations; lean on Stage-2 + calibration.
3. **Effective resolution off-ALS is ~10 m + synthesized texture**, not measured 1 m — state plainly.
4. **Absolute calibration** still needs co-located destructive truth (RxCADRE clip plots, manual download).

See [TODO.md](../TODO.md) for the ordered build plan and [IDEAS.md](../IDEAS.md) for strategy.
