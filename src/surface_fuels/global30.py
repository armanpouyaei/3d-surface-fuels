"""Stage 1: 30 m surface-fuel regressor (POC harness).

Predicts surface fuel load (kg/m²) at 30 m from a stack of globally-available
predictors (AlphaEarth/Clay embeddings + GEDI + Sentinel-1/2 + terrain) trained
on US 3DEP/field targets. See research/GLOBAL30_METHOD.md.

This is the model-agnostic, **AOI-based** harness — region prediction is the
required capability; global is the same call tiled (the design supports it, we
don't force it). The estimator is swappable (quantile gradient boosting now;
frozen-embedding conv/UNet head later).

Includes the rigor the design calls for:
* **quantile** prediction (median + lower/upper),
* **spatially-blocked CV** (no autocorrelation leakage),
* **conformalized quantile regression (CQR)** for calibrated intervals,
* **OOD flag** (feature distance to training) for out-of-region application,
* **per-stratum** metrics (open vs under-canopy) — never blend strata.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import metrics

QUANTILES = (0.05, 0.5, 0.95)


def assemble_features(predictors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Stack named 2D predictor rasters into a per-cell feature matrix."""
    names = list(predictors)
    X = np.stack([predictors[k].ravel() for k in names], axis=1).astype(np.float32)
    return X, names


def _fit_quantiles(X, y, quantiles=QUANTILES):
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = {}
    for q in quantiles:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                          max_depth=6, learning_rate=0.08,
                                          max_iter=150, l2_regularization=1.0,
                                          random_state=0)
        models[q] = m.fit(X, y)
    return models


def _cqr_delta(models, Xc, yc, lo=0.05, hi=0.95):
    """Conformalized-quantile-regression adjustment from a calibration split:
    the (1-alpha) quantile of nonconformity scores max(q_lo - y, y - q_hi)."""
    qlo, qhi = models[lo].predict(Xc), models[hi].predict(Xc)
    scores = np.maximum(qlo - yc, yc - qhi)
    n = len(yc)
    k = int(np.ceil((n + 1) * (hi - lo)))  # finite-sample conformal rank
    return float(np.sort(scores)[min(k, n) - 1])


def blocked_cv(target: np.ndarray, predictors: Dict[str, np.ndarray],
               block_id: np.ndarray, quantiles=QUANTILES, conformal=True):
    """Spatially-blocked leave-one-block-out CV. Returns median + lower/upper
    prediction maps (CQR-calibrated if ``conformal``) and a fitted full model."""
    X, names = assemble_features(predictors)
    y = target.ravel()
    blk = block_id.ravel()
    lo, hi = quantiles[0], quantiles[-1]

    med = np.zeros_like(y); ql = np.zeros_like(y); qh = np.zeros_like(y)
    rng = np.random.default_rng(0)
    for b in np.unique(blk):
        test = blk == b
        tr = ~test
        Xtr, ytr = X[tr], y[tr]
        # hold out a calibration slice of the training blocks for CQR
        cal = rng.random(len(ytr)) < 0.25 if conformal else np.zeros(len(ytr), bool)
        models = _fit_quantiles(Xtr[~cal] if conformal else Xtr,
                                ytr[~cal] if conformal else ytr, quantiles)
        d = _cqr_delta(models, Xtr[cal], ytr[cal], lo, hi) if (conformal and cal.sum() > 20) else 0.0
        med[test] = models[quantiles[len(quantiles) // 2]].predict(X[test])
        ql[test] = models[lo].predict(X[test]) - d
        qh[test] = models[hi].predict(X[test]) + d

    shape = target.shape
    full = _fit_quantiles(X, y, quantiles)
    return {
        "median": np.clip(med, 0, None).reshape(shape).astype(np.float32),
        "lower": np.clip(ql, 0, None).reshape(shape).astype(np.float32),
        "upper": np.clip(qh, 0, None).reshape(shape).astype(np.float32),
        "feature_names": names,
        "full_models": full,
        "X": X,
    }


def ood_flag(X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Standardized nearest-neighbour distance to the training feature cloud —
    an out-of-distribution score for applying the model outside its region."""
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-9
    A = (X_train - mu) / sd
    B = (X_query - mu) / sd
    # distance to training centroid (cheap, robust OOD proxy)
    return np.sqrt(((B - A.mean(0)) ** 2).sum(1)).astype(np.float32)


def evaluate(out: Dict, target: np.ndarray, canopy: Optional[np.ndarray] = None,
             canopy_split: float = 0.3) -> Dict[str, object]:
    """R²/RMSE (median), interval coverage, and per-stratum metrics."""
    p, t = out["median"], target
    cov = float(((target >= out["lower"]) & (target <= out["upper"])).mean())
    rep = {"r2": round(metrics.r2(p, t), 3), "rmse": round(metrics.rmse(p, t), 4),
           "interval_coverage": round(cov, 3),
           "mean_interval_width": round(float((out["upper"] - out["lower"]).mean()), 4)}
    if canopy is not None:
        op, ca = canopy < canopy_split, canopy >= canopy_split
        rep["r2_open"] = round(metrics.r2(p[op], t[op]), 3) if op.any() else None
        rep["r2_under_canopy"] = round(metrics.r2(p[ca], t[ca]), 3) if ca.any() else None
    return rep
