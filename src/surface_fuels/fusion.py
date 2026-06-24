"""SAR + LiDAR fusion for surface-fuel load (Week 2).

The method is deliberately simple and explainable:

    fused = (1 - w) * lidar_estimate  +  w * sar_estimate,   w = canopy cover

LiDAR understory returns are trusted where the canopy is open (w -> 0); the
canopy-penetrating SAR signal takes over where the overstory occludes the LiDAR
(w -> 1). Each sensor is first calibrated to fuel load by a simple linear fit
(LiDAR calibrated on open cells, SAR on all cells). Validated by spatially
blocked cross-validation so no test cell shares a neighbourhood with training.

This implements approach "A->B" from IDEAS.md and directly demonstrates that
radar fills LiDAR's under-canopy blind spot.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import metrics

CANOPY_SPLIT = 0.3  # cells with cover >= this are "under canopy" for stratified metrics


def _linfit(x, y):
    """Least-squares a*x + b; returns (a, b)."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def fuse_predict(scene, train: np.ndarray):
    """Calibrate + blend using only ``train`` cells. Returns (lidar, sar, fused)
    estimate maps over the whole grid."""
    L = scene.truth_load
    li, sr, C = scene.lidar_obs, scene.sar_obs, scene.canopy_obs

    open_train = train & (C < CANOPY_SPLIT)
    if open_train.sum() < 25:           # ensure enough open cells to calibrate LiDAR
        open_train = train
    a_l, b_l = _linfit(li[open_train], L[open_train])  # LiDAR calibrated where it works
    a_s, b_s = _linfit(sr[train], L[train])            # SAR calibrated everywhere

    lidar_est = np.clip(a_l * li + b_l, 0, None)
    sar_est = np.clip(a_s * sr + b_s, 0, None)
    w = np.clip(scene.canopy_obs, 0, 1)                # blend weight = canopy cover
    fused = np.clip((1 - w) * lidar_est + w * sar_est, 0, None)
    return lidar_est.astype(np.float32), sar_est.astype(np.float32), fused.astype(np.float32)


def spatial_block_cv(scene, blocks: int = 4) -> Dict[str, np.ndarray]:
    """Spatially blocked CV: hold out each block, fit on the rest, predict the
    held-out block. Returns full-grid prediction maps for each method."""
    ny, nx = scene.truth_load.shape
    by = (np.arange(ny) * blocks // ny)
    bx = (np.arange(nx) * blocks // nx)
    block_id = by[:, None] * blocks + bx[None, :]

    out = {k: np.zeros_like(scene.truth_load) for k in ("lidar_only", "sar_only", "fusion")}
    for b in np.unique(block_id):
        test = block_id == b
        l_est, s_est, f_est = fuse_predict(scene, ~test)
        out["lidar_only"][test] = l_est[test]
        out["sar_only"][test] = s_est[test]
        out["fusion"][test] = f_est[test]
    return out


def _stats(pred, L, C):
    op, ca = C < CANOPY_SPLIT, C >= CANOPY_SPLIT
    return {
        "r2": round(metrics.r2(pred, L), 3),
        "rmse": round(metrics.rmse(pred, L), 4),
        "r2_open": round(metrics.r2(pred[op], L[op]), 3),
        "r2_under_canopy": round(metrics.r2(pred[ca], L[ca]), 3),
        "rmse_under_canopy": round(metrics.rmse(pred[ca], L[ca]), 4),
    }


def evaluate(pred: Dict[str, np.ndarray], scene) -> Dict[str, Dict[str, float]]:
    """Per-method metrics vs truth, including the FastFuels uniform baseline,
    stratified by open vs. under-canopy."""
    L, C = scene.truth_load, scene.canopy
    report = {name: _stats(p, L, C) for name, p in pred.items()}
    report["fastfuels_uniform"] = _stats(scene.uniform_load, L, C)
    return report
