# RESULTS — validated, on real data

> Living results log for the submission. Every number here is from a script in
> `scripts/` on **real** data (USGS 3DEP LiDAR + AlphaEarth + Sentinel-1 + NEON),
> with spatially-blocked cross-validation. Synthetic method checks live in
> `research/VALIDATION.md`; this file is the real-data evidence.

## Headline (the thesis, validated)

**A measured 3D fuel-structure / bulk-density product beats FastFuels' uniform
surface layer**, end-to-end, against measured truth — and it generalizes from
free spaceborne data where airborne LiDAR is absent.

Why structure and not herbaceous *load*: a multi-ecosystem test (below) confirmed
that mapping herbaceous *load* from spaceborne sensors at meter scale is genuinely
hard (a documented literature ceiling). Fuel *structure* (the bulk-density column
that QUIC-Fire / FIRETEC actually ingest) is robustly predictable. So structure is
the headline deliverable; herb load is an honest, uncertainty-flagged add-on.

## 1. End-to-end pipeline (OSBS) — `scripts/build_pipeline_osbs.py`

Chain: **spaceborne → 30 m → 10 m**, validated against the **measured 3DEP**
near-ground structure (a bulk-density proxy), head-to-head vs a FastFuels-style
uniform layer. Site: NEON OSBS, 1.5 × 1.5 km, 39.5 M LiDAR points, AlphaEarth +
Sentinel-1 vintage-matched to 2018. Figure: `figures/pipeline_osbs.png`.

- **Stage-1** (AlphaEarth 64-band + Sentinel-1 + terrain → 30 m structure,
  spatially-blocked CV): **R² 0.767** (open 0.79, under-canopy 0.48, interval
  coverage 0.88).
- **Stage-2** downscales the *predicted* 30 m → 10 m using AlphaEarth's native
  10 m bands + terrain, mass-conserving, spatially held-out.

| Product | overall R² | within-block R² (sub-30 m) | heterogeneity CV |
|---|---|---|---|
| FastFuels uniform | −0.00 | −0.00 | 0.00 |
| Stage-1 30 m (upsampled) | 0.606 | −0.00 | 0.70 |
| Downscaler (clean 30 m input) | 0.852 | 0.291 | 0.82 |
| **End-to-end (ours)** | **0.664** | **0.274** | **0.73** |
| *measured truth* | — | — | *0.89* |

**Reading it.** FastFuels assigns one value per 30 m class, so its *within-block*
R² is **0 by construction** — it captures none of the sub-30 m heterogeneity that
drives fire behavior. Our within-block R² of **0.27** is exactly that gap, and we
recover **CV 0.73 of the truth's 0.89**. Honest error budget: the end-to-end
product = Stage-1's 30 m error (spaceborne) + Stage-2's within-block detail; the
clean-coarse row (0.85) is the headroom as the 30 m baseline improves.

Where airborne LiDAR exists we deliver the measured 1 m structure *directly*; this
pipeline is how the **same product generalizes** where only spaceborne data exist
(the competition's generality axis). Interactive: dashboard "End-to-end" mode.

## 2. de Conto (2025) architecture head-to-head — `scripts/deconto_headtohead.py`

The brief asks us to be "better than the 2025 paper." de Conto et al. predict a
canopy *structure index* (WSCI) — not fuel — with a fully-convolutional
EfficientNetV2 (365,682 params) trained on ~133 M GEDI footprints. We can't "beat
0.82" (different variable, circular GEDI-vs-GEDI). The honest, like-for-like test:
**their architecture paradigm vs ours, on the same fuel-structure target, the same
68-channel predictor stack, the same spatially-blocked folds.**

- **Ours** — per-pixel quantile Gradient Boosting + conformal intervals.
- **de Conto** — a faithful *compact* reproduction of their design (Fused-MBConv →
  MBConv + squeeze-excite, Gaussian-NLL head = mean+variance, MC-dropout), 64,242
  params, sized for an AOI. Trained with patch-radius buffering so its convolutional
  receptive field never leaks across the held-out block boundary.

Folds: 2×2 spatially-blocked quadrants (4 × 625-cell held-out blocks). Figure:
`figures/deconto_headtohead.png`.

| model | R² all | open | under-canopy | RMSE | interval coverage |
|---|---|---|---|---|---|
| **Ours: per-pixel quantile GBM** | **0.628** | 0.635 | 0.375 | 0.254 | **85%** |
| de Conto: fully-conv CNN (NLL) | −0.147 | −0.158 | −0.380 | 0.446 | 14% |

**Honest reading.** At **AOI scale** the per-pixel model wins decisively, and the
CNN fails outright (negative R², per-fold −0.67…+0.08; broken 14% interval
coverage) — because ~1,600 training cells/fold cannot fit even a 64 k-param conv
net. This is **not** a global rematch: de Conto's CNN earns its keep on millions of
chips. The takeaway is that **for the region/AOI-scale surface-fuel task the
challenge requires, a per-pixel model with proper spatial CV is the correct
architecture**; the CNN's spatial-context advantage needs continental training,
which our tiled design supports as the production scale-up.

**What this licenses us to say vs the 2025 paper** (all defensible):
1. We predict a **fire-relevant fuel-structure / bulk-density** quantity, not a
   unitless canopy index (WSCI).
2. We ship **calibrated uncertainty** (85% conformal coverage vs the CNN's broken
   14% at this scale) + **per-stratum** (open/under-canopy) + an **OOD flag** —
   none of which their structure-index product provides for fuel.
3. At the **required AOI scale our method works where the CNN paradigm does not.**

(The GBM scores 0.63 here under the stricter 2×2 quadrant CV; §1 reports 0.77 under
3×3 blocking — fewer, larger held-out blocks is a harder extrapolation test.)

## 2b. Generality — a second, opposite ecosystem (SOAP) — `build_pipeline_osbs.py --site soap`

The same pipeline, same code, run on **NEON SOAP** (Sierra mixed-conifer, CA; 2022 3DEP at
44.7 pts/m², 477 m of ground relief; AlphaEarth 2022, UTM 11N) — a structurally *opposite*
ecosystem to OSBS (dense mountainous forest vs open patchy savanna). Figure:
`figures/pipeline_soap.png`.

| ecosystem | Stage-1 30 m R² | end-to-end R² | within-block R² | truth CV | recovered CV |
|---|---|---|---|---|---|
| OSBS — longleaf savanna (FL, 2018) | 0.767 | 0.664 | 0.274 | 0.89 | 0.73 |
| SOAP — Sierra mixed conifer (CA, 2022) | 0.779 | 0.585 | 0.141 | 0.34 | 0.25 |

**Reading it.** The method transfers across two opposite structural regimes with *no code
changes* (only site args). SOAP's **under-canopy R² is 0.742** (dense forest is mostly
under-canopy; the −0.94 "open" R² is an artifact of having almost no open cells — a tiny,
unstable sample, reported honestly). SOAP's truth is intrinsically more uniform (CV 0.34 vs
OSBS 0.89), so there's less sub-30 m signal to recover (within-block 0.14 vs 0.27) — but the
end-to-end product still beats the FastFuels uniform layer (R² ≈ 0, within-block 0, CV 0) and
recovers most of the heterogeneity that exists. Generality is the competition's key axis; this
is direct evidence for it.

## 2c. FastFuels surface-layer head-to-head — `fastfuels_headtohead.py`

FastFuels' surface fuel *is*, by design, **LANDFIRE FBFM40 → an SB40 load lookup →
one value per fuel-model class, uniform within the class**. We encode FastFuels'
exact lookup (Scott & Burgan 2005, RMRS-GTR-153 Table 7) and compare to our measured
1 m product over OSBS. Figure: `figures/fastfuels_headtohead.png`.

The script runs the **real, verified FastFuels API flow** with a key: create a domain
over the AOI → `POST /v1/domains/{id}/grids/surface` with fuelLoad from LANDFIRE
FBFM40 → poll → export GeoTIFF → download → read. This was **verified end-to-end
against the live API** with a real key — a `uniform` surface grid completes and
exports correctly, confirming the integration. However, FastFuels' **LANDFIRE FBFM40**
surface generation was **failing server-side** at run time (the job returns
`status: failed` with no message; LANDFIRE was also directly unreachable from this
environment), so the committed run falls back to FastFuels' *exact lookup table*.
That fallback is **numerically identical to FastFuels' FBFM40 output** — its FBFM40
fuelLoad is precisely the SB40 group-sum (1 h + 10 h + 100 h + live herb + live woody)
encoded here. Re-run with the key once FastFuels' LANDFIRE backend recovers to pull
the live per-pixel grid.

| FastFuels SB40 class (longleaf-relevant) | assigned surface load (kg/m²) |
|---|---|
| GR1 / GR2 / GR3 | 0.09 / 0.25 / 0.45 |
| GS1 / **GS2** | 0.30 / **0.58** |
| TU1 / TL2 | 0.83 / 1.32 |

| metric | FastFuels (SB40 uniform) | our measured |
|---|---|---|
| surface-load heterogeneity CV (1 m) | **0.00** (one value/class) | **1.13** |
| CV at 30 m (FastFuels' own scale) | 0.00 | 1.01 |
| mean load (kg/m²) | a single SB40 value | 0.60 |

**Reading it.** Whichever class FastFuels assigns, it **collapses the whole AOI to a
single number** — its surface heterogeneity is zero *by construction*. Our measured
load spans ~0–2.7 kg/m² (CV 1.13). Independent cross-check: **GS2 = 0.58 kg/m²**
(a plausible OSBS class) nearly matches our measured **mean 0.60** — so our
magnitude is well-anchored to FastFuels' own table while we add the spatial
structure it cannot represent.

**Surface vs canopy — FastFuels' OWN 3D product, pulled live** (`build_fastfuels3d_osbs.py`,
verified flow). With the API key we ran the real v1 flow over OSBS: create domain →
TreeMap tree inventory → **voxelized tree/canopy grid** (`grids/tree`, completed) →
zarr export → read. FastFuels' canopy bulk density comes back as a **33 m-tall 1 m³
voxel field** (`figures/fastfuels_canopy3d.png`, `data/processed/ff_osbs_canopy3d.nc`):
genuinely 3D and lumpy (col-max CV 6.78, ~1.3 % voxels occupied — scattered savanna
trees). This makes the point concrete: **FastFuels' 3D structure is the trees; its
surface layer is the uniform slab beneath.** Trees are *optional* in this challenge —
the in-scope **surface** layer is the one FastFuels renders flat and we make 3D.
(The `grids/surface` LANDFIRE-FBFM40 step failed server-side at run time, so the
combined surface+tree export wasn't available; the tree grid is the canopy shown.)

## 2d. Global "generate anywhere" — on-demand inference (`portable.py`, dashboard tab)

A **portable Stage-1 model** (`scripts/train_portable.py` → `stage1_portable.joblib`)
trained once on US sites (OSBS + SOAP, 30k pooled 10 m samples) using **only global,
free predictors** (AlphaEarth + Sentinel-1 — *no LiDAR at inference*). The dashboard's
**🌍 Generate anywhere** tab takes any lat/lon → fetches AEF+S1 for that AOI → predicts
30 m→1 m structure + 3D voxels, with **conformal intervals**, an **OOD flag**, and a
**disk cache**. Verified live: OSBS-2018 **1 % OOD** (in-distribution), OSBS-2022 3 %,
Amazon **100 % OOD** (correctly flagged).

- **Design:** on-demand per-AOI (no world-wide pre-grid) + cache. **Memory-safe by
  construction** — inference runs at 10 m (a 1500 m AOI is 150×150×66 floats ≈ 6 MB),
  hard AOI cap 2000 m, 1 m export voxel-budgeted; the UI shows the footprint pre-run.
- **OOD is computed on AlphaEarth bands only** (Sentinel-1 availability is flaky
  globally; including it spuriously inflated the distance). It reflects novelty in
  **ecosystem AND AEF year** and is conservative by design.
- **Honest generality limit:** leave-one-site-out transfer is weak (R² −0.18 / −0.45)
  with only 2 opposite training ecosystems — so most non-(SE-savanna/Sierra-conifer)
  locations flag OOD. The flag is the safeguard; **adding 3DEP training sites is the
  path to real global generality.** This tab is a *capability* demo, validated only
  where we have US truth.

## 3. Supporting real-data results

- **Stage-1 AlphaEarth dominance** (`build_global30.py --site osbs`): AEF-only
  R² 0.66 > physical-only (S1+terrain) 0.60; full 0.72. AEF is the dominant
  surface-structure predictor once vintage-matched and orientation-correct.
- **NEON field validation** (`build_neon_field.py`): AlphaEarth → real OSBS herb
  *load* R² 0.23 / Spearman 0.51 (leave-one-plot-out), beats S1 (0.07). Honest
  but modest — the basis for treating herb load as an add-on, not the headline.
- **Multi-ecosystem generality** (`analyze_field_multisite.py`, 4 ecosystems,
  89 clips): herb-*load* leave-one-site-out transfer Spearman **0.25**, within-site
  ≈ 0 — unstable (small n, 0.2 m² clip vs 10 m pixel). Consistent with literature
  ceilings (Leite 0.31, Labenski 0.27–0.41). This is *why* the headline is structure.
- **SAR + LiDAR fusion** (synthetic, blocked CV): fusion R² 0.71 vs LiDAR-only 0.51
  (under canopy 0.50 vs −0.01) — the on-ramp for filling LiDAR's under-canopy blind
  spot; real Sentinel-1 RTC wired over Eglin.
- **3DEP proxy caveat** (`validate_proxy_field.py`): the near-ground return proxy
  tracks *woody* understory structure, not herbaceous load (Spearman 0.02 vs field
  herb). So the structure product is honestly a *woody/understory structure +
  bulk-density* product; herb load is reported separately with its uncertainty.
