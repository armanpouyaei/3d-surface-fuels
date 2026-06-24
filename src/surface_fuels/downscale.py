"""30 m -> 1 m surface-fuel downscaler (Week 3 POC).

The global-product idea: a coarse (~30 m) surface-fuel baseline is available
*everywhere* from spaceborne sensors (GEDI + Sentinel-1/NISAR + Sentinel-2);
airborne-LiDAR (3DEP) 1 m fuel grids exist only over the US. So we learn a
**downscaler** `f([coarse 30 m fuel, fine 10 m covariates]) -> 1 m fuel` where
the US provides the 1 m *target* and the *inputs* are globally available — then
apply it anywhere the coarse baseline + covariates exist.

This module is the regression POC. It is deliberately model-agnostic: the
feature stack + mass-conservation + blocked-CV harness stay fixed, and the
estimator is swappable (RandomForest now; a Clay/AlphaEarth-embedding UNet
later — see research/DOWNSCALING.md).

Key properties:
* **Mass-conserving** — the predicted 1 m field is rescaled so each coarse block
  averages back to the coarse baseline value (a true disaggregation, not free
  invention).
* **Honest validation** — spatially-blocked CV (quadrant holdout), judged on
  *within-block* skill (the sub-30 m detail a naive upsample cannot produce) and
  on heterogeneity recovery (CV), not just pixelwise error.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import metrics


# ── coarse/fine grid ops ─────────────────────────────────────────────────────
def block_coarsen(field: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a fine field to coarse resolution (the spaceborne baseline)."""
    ny, nx = field.shape
    ny2, nx2 = ny // factor, nx // factor
    return field[:ny2 * factor, :nx2 * factor].reshape(ny2, factor, nx2, factor).mean(axis=(1, 3))


def upsample(coarse: np.ndarray, factor: int, shape: Tuple[int, int]) -> np.ndarray:
    """Block-replicate a coarse field back to fine resolution (naive baseline)."""
    up = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
    return up[:shape[0], :shape[1]]


def block_anomaly(fine: np.ndarray, factor: int) -> np.ndarray:
    """Fine field minus its own coarse block-mean — the sub-30 m signal a
    covariate carries (what lets the model place within-block texture)."""
    return fine - upsample(block_coarsen(fine, factor), factor, fine.shape)


def enforce_mass_conservation(pred: np.ndarray, coarse: np.ndarray, factor: int) -> np.ndarray:
    """Rescale each coarse block so the 1 m mean matches the coarse baseline."""
    out = pred.copy()
    ny2, nx2 = coarse.shape
    for j in range(ny2):
        for i in range(nx2):
            blk = out[j * factor:(j + 1) * factor, i * factor:(i + 1) * factor]
            m = blk.mean()
            if m > 1e-9:
                blk *= coarse[j, i] / m
    return np.clip(out, 0, None)


# ── feature stack (swap-friendly: add embedding channels here later) ─────────
def stack_features(coarse_up: np.ndarray, covars: Dict[str, np.ndarray],
                   factor: int) -> Tuple[np.ndarray, List[str]]:
    """Per-cell feature matrix from the coarse baseline + fine covariates and
    their sub-block anomalies. ``covars`` maps name -> fine (ny, nx) array.
    (A future model would append Clay/AlphaEarth embedding channels here.)"""
    feats = [coarse_up]
    names = ["coarse_fuel"]
    for k, v in covars.items():
        feats.append(v)
        names.append(k)
        feats.append(block_anomaly(v, factor))
        names.append(f"{k}_subblock")
    X = np.stack([f.ravel() for f in feats], axis=1).astype(np.float32)
    return X, names


def make_estimator():
    """Default regression estimator for the POC (swap for a UNet later)."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=8,
                                 n_jobs=-1, random_state=0)


# ── quadrant cross-validation ────────────────────────────────────────────────
def _quadrant_mask(ny, nx, k):
    """Boolean test mask for quadrant k in {0,1,2,3} (spatially separated)."""
    my, mx = ny // 2, nx // 2
    rows = slice(0, my) if k < 2 else slice(my, ny)
    cols = slice(0, mx) if k % 2 == 0 else slice(mx, nx)
    m = np.zeros((ny, nx), bool)
    m[rows, cols] = True
    return m


def downscale_cv(target: np.ndarray, coarse: np.ndarray, covars: Dict[str, np.ndarray],
                 factor: int, mass_conserve: bool = True, estimator_factory=make_estimator):
    """Spatially-blocked (quadrant) CV. Returns prediction maps + the coarse
    baseline + a fitted estimator (on all data, for inspection/feature importance)."""
    ny, nx = target.shape
    coarse_up = upsample(coarse, factor, target.shape)
    X, names = stack_features(coarse_up, covars, factor)
    y = target.ravel()

    pred = np.zeros_like(target).ravel()
    flat_q = np.concatenate([_quadrant_mask(ny, nx, k).ravel()[None] for k in range(4)])
    for k in range(4):
        test = flat_q[k]
        est = estimator_factory()
        est.fit(X[~test], y[~test])
        pred[test] = est.predict(X[test])
    pred = np.clip(pred.reshape(ny, nx), 0, None)
    if mass_conserve:
        pred = enforce_mass_conservation(pred, coarse, factor)

    full = estimator_factory().fit(X, y)
    return {
        "downscaled": pred.astype(np.float32),
        "coarse_baseline": coarse_up.astype(np.float32),
        "feature_names": names,
        "feature_importance": getattr(full, "feature_importances_", None),
    }


# ── metrics ──────────────────────────────────────────────────────────────────
def evaluate(out: Dict, target: np.ndarray, factor: int) -> Dict[str, Dict[str, float]]:
    """Compare the downscaler and the naive coarse upsample against the 1 m truth.

    ``within_block_r2`` removes each method's coarse block-mean first, isolating
    the sub-30 m detail — the coarse upsample is flat within a block so it scores
    ~0 there; the downscaler's score is the real downscaling skill.
    """
    def block_demean(a):
        return a - upsample(block_coarsen(a, factor), factor, a.shape)

    def stats(p):
        cv = float(p.std() / (p.mean() + 1e-9))
        agg = block_coarsen(p, factor)
        coarse_truth = block_coarsen(target, factor)
        return {
            "r2": round(metrics.r2(p, target), 3),
            "rmse": round(metrics.rmse(p, target), 4),
            "within_block_r2": round(metrics.r2(block_demean(p), block_demean(target)), 3),
            "cv": round(cv, 3),
            "agg_consistency_r2": round(metrics.r2(agg, coarse_truth), 3),
        }

    return {
        "downscaler": stats(out["downscaled"]),
        "coarse_baseline": stats(out["coarse_baseline"]),
        "truth_cv": round(float(target.std() / (target.mean() + 1e-9)), 3),
    }
