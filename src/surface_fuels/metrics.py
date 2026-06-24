"""Validation metrics for comparing a predicted fuel grid against truth.

The challenge weights *validation credibility* most heavily, so these metrics
cover three complementary questions:

1. **Mass accuracy** — do we get the fuel *amount* right? (RMSE / bias / R^2 on
   2D fuel load in kg/m^2).
2. **Structure accuracy** — do we get the *3D arrangement* right? (voxel
   occupancy IoU, vertical-profile agreement).
3. **Heterogeneity** — do we reproduce the *spatial variability* that uniform
   fuel models destroy? (coefficient of variation, spatial autocorrelation).

All functions operate on `FuelVoxelGrid` objects or plain numpy arrays.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .voxel import FuelVoxelGrid


# -- scalar error metrics ----------------------------------------------
def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def bias(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred - truth))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def r2(pred: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, float).ravel()
    pred = np.asarray(pred, float).ravel()
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# -- composite comparisons ---------------------------------------------
def fuel_load_metrics(pred: FuelVoxelGrid, truth: FuelVoxelGrid) -> Dict[str, float]:
    """Accuracy of 2D fuel loading (kg/m^2)."""
    p, t = pred.fuel_load(), truth.fuel_load()
    return {
        "load_rmse_kg_m2": round(rmse(p, t), 4),
        "load_bias_kg_m2": round(bias(p, t), 4),
        "load_mae_kg_m2": round(mae(p, t), 4),
        "load_r2": round(r2(p, t), 4),
    }


def occupancy_iou(pred: FuelVoxelGrid, truth: FuelVoxelGrid,
                  threshold: float = 0.05) -> float:
    """Intersection-over-union of occupied voxels (3D structure agreement)."""
    p = pred.occupancy(threshold)
    t = truth.occupancy(threshold)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / (union + 1e-12))


def vertical_profile_metrics(pred: FuelVoxelGrid,
                             truth: FuelVoxelGrid) -> Dict[str, float]:
    """Agreement of the mean vertical bulk-density profile."""
    p, t = pred.vertical_profile(), truth.vertical_profile()
    return {
        "profile_rmse_kg_m3": round(rmse(p, t), 4),
        "profile_r2": round(r2(p, t), 4),
    }


def heterogeneity_stats(grid: FuelVoxelGrid) -> Dict[str, float]:
    """Spatial-variability descriptors of the 2D fuel-load field.

    ``moran_i`` is a rook-neighbour spatial autocorrelation index in [-1, 1];
    higher means stronger clumping. Uniform fuel models collapse CV -> 0.
    """
    load = grid.fuel_load()
    mean = float(load.mean())
    std = float(load.std())
    return {
        "mean_load_kg_m2": round(mean, 4),
        "std_load_kg_m2": round(std, 4),
        "cv": round(std / (mean + 1e-9), 4),
        "moran_i": round(_morans_i(load), 4),
    }


def _morans_i(field: np.ndarray) -> float:
    """Rook-contiguity Moran's I for a 2D field (simple, dependency-free)."""
    z = field - field.mean()
    denom = np.sum(z ** 2)
    if denom <= 0:
        return 0.0
    # sum of products of adjacent cells (right + down neighbours, counted twice)
    num = (
        np.sum(z[:, :-1] * z[:, 1:])
        + np.sum(z[:-1, :] * z[1:, :])
    ) * 2.0
    n = field.size
    # number of adjacency pairs (each counted twice to match num)
    w = (field.shape[0] * (field.shape[1] - 1)
         + (field.shape[0] - 1) * field.shape[1]) * 2.0
    return float((n / w) * (num / denom))


def bulk_density_rmse_3d(pred: FuelVoxelGrid, truth: FuelVoxelGrid) -> float:
    """Per-voxel bulk-density RMSE (kg/m^3) over the whole 3D grid."""
    return round(rmse(pred.bulk_density, truth.bulk_density), 4)


def depth_rmse(pred: FuelVoxelGrid, truth: FuelVoxelGrid) -> float:
    """RMSE of 2D fuelbed depth (m)."""
    return round(rmse(pred.fuelbed_depth(), truth.fuelbed_depth()), 4)


def compare(pred: FuelVoxelGrid, truth: FuelVoxelGrid,
            threshold: float = 0.05) -> Dict[str, float]:
    """Full validation report comparing ``pred`` to ``truth``."""
    out: Dict[str, float] = {}
    out.update(fuel_load_metrics(pred, truth))
    out.update(vertical_profile_metrics(pred, truth))
    out["bulk_density_rmse_3d_kg_m3"] = bulk_density_rmse_3d(pred, truth)
    out["depth_rmse_m"] = depth_rmse(pred, truth)
    out["occupancy_iou"] = round(occupancy_iou(pred, truth, threshold), 4)
    out["total_mass_pred_kg"] = round(pred.total_mass(), 1)
    out["total_mass_truth_kg"] = round(truth.total_mass(), 1)
    out["mass_error_pct"] = round(
        100.0 * (pred.total_mass() - truth.total_mass())
        / (truth.total_mass() + 1e-9), 2)
    # how well is heterogeneity preserved (ratio of CVs; 1.0 == perfect)
    het_p = heterogeneity_stats(pred)
    het_t = heterogeneity_stats(truth)
    out["cv_pred"] = het_p["cv"]
    out["cv_truth"] = het_t["cv"]
    out["cv_ratio"] = round(het_p["cv"] / (het_t["cv"] + 1e-9), 4)
    return out
