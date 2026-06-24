"""Synthetic surface-fuel scenes for the MVP demo and validation harness.

These generators let us build and test the full pipeline (3D voxels -> property
maps -> validation -> dashboard) *before* wiring in real LiDAR/SAR data. The
scene is modelled on a **longleaf pine / wiregrass** surface fuelbed — the
canonical SERDP / Tall Timbers system where surface fuels are famously clumpy at
sub-metre scale — using literature-plausible loads and depths.

Three grids are produced for the headline comparison:

* ``truth``      — the real, heterogeneous, clumpy surface fuelbed.
* ``fastfuels``  — what a *categorical* fuel-model product (FastFuels / LANDFIRE
                   SB40) produces: a single fuelbed applied **uniformly**, i.e.
                   the domain-mean vertical profile in every column.
* ``ours``       — heterogeneity **recovered** from a simulated remote-sensing
                   texture predictor (a stand-in for LiDAR understory returns /
                   SAR backscatter), then re-voxelized.

The point of the demo: ``ours`` reproduces the clumping (and beats ``fastfuels``
on load RMSE / R^2) while ``fastfuels`` collapses all horizontal variability.

Plausible value ranges used (longleaf surface fuels):
  litter (pine needle) load ~0.2-1.0 kg/m^2; wiregrass ~0.1-0.6 kg/m^2;
  shrub (gallberry/palmetto) patches up to ~2 kg/m^2 and ~1-3 m tall;
  total surface load typically ~0.3-1.5 kg/m^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter

from .voxel import FuelVoxelGrid, GeoRef

# SB40-style component priors (used for composition arrays we don't "measure").
SAV_LITTER = 2000.0   # m^2/m^3  (pine litter)
SAV_GRASS = 3500.0    # m^2/m^3  (fine herbaceous)
SAV_SHRUB = 1600.0    # m^2/m^3  (woody shrub foliage)
LIVE_FRAC_LITTER = 0.0
LIVE_FRAC_GRASS = 0.45    # dormant-season wiregrass: mostly cured
LIVE_FRAC_SHRUB = 0.75


def _gaussian_field(shape, scale: float, rng: np.random.Generator) -> np.ndarray:
    """A spatially-correlated random field in ~[0, 1] (white noise + blur)."""
    f = gaussian_filter(rng.standard_normal(shape), sigma=scale, mode="reflect")
    f -= f.min()
    f /= f.max() + 1e-12
    return f


def _distribute(load2d: np.ndarray, height2d: np.ndarray, nz: int,
                dz: float = 1.0) -> np.ndarray:
    """Distribute a 2D load (kg/m^2) over height into a 3D bulk-density grid.

    A column with load ``L`` and fuelbed height ``H`` spreads its mass evenly
    over ``n = ceil(H/dz)`` voxels, giving each a bulk density of
    ``(L / n) / dz`` so the column integrates back to ``L``.
    """
    ny, nx = load2d.shape
    bd = np.zeros((nz, ny, nx), dtype=np.float32)
    n_vox = np.clip(np.ceil(height2d / dz).astype(int), 0, nz)
    for k in range(1, nz + 1):
        mask = n_vox == k
        if not mask.any():
            continue
        per_voxel = (load2d[mask] / k) / dz
        for z in range(k):
            bd[z][mask] += per_voxel
    return bd


@dataclass
class DemoScene:
    truth: FuelVoxelGrid
    fastfuels: FuelVoxelGrid
    ours: FuelVoxelGrid
    # 2D fields kept for plotting / debugging
    truth_load: np.ndarray
    fastfuels_load: np.ndarray
    ours_load: np.ndarray


def _component_fields(nx: int, ny: int, rng: np.random.Generator):
    """Build litter / grass / shrub load & height fields for a longleaf scene."""
    shape = (ny, nx)

    # 1) Litter: fairly continuous, gently varying needle bed.
    litter = 0.25 + 0.55 * _gaussian_field(shape, scale=6.0, rng=rng)  # kg/m^2
    litter_h = np.full(shape, 0.05)  # ~5 cm bed -> bottom voxel

    # 2) Wiregrass: clumpy bunchgrass. Threshold a fine field into bunches.
    g = _gaussian_field(shape, scale=2.0, rng=rng)
    grass_mask = g > 0.55
    grass = np.where(grass_mask, 0.15 + 1.0 * (g - 0.55) / 0.45, 0.0)  # kg/m^2
    grass_h = np.where(grass_mask, 0.4 + 0.4 * g, 0.0)  # 0.4-0.8 m

    # 3) Shrub patches (gallberry/palmetto): sparse, taller, denser.
    s = _gaussian_field(shape, scale=4.0, rng=rng)
    shrub_mask = s > 0.78
    shrub = np.where(shrub_mask, 0.5 + 2.0 * (s - 0.78) / 0.22, 0.0)  # kg/m^2
    shrub_h = np.where(shrub_mask, 1.0 + 2.0 * (s - 0.78) / 0.22, 0.0)  # 1-3 m

    return {
        "litter": (litter, litter_h),
        "grass": (grass, grass_h),
        "shrub": (shrub, shrub_h),
    }


def make_truth(nx: int = 100, ny: int = 100, nz: int = 4, seed: int = 42,
               crs: str = "EPSG:32616", x0: float = 500000.0,
               y0: float = 3350000.0) -> FuelVoxelGrid:
    """Generate a heterogeneous longleaf-pine surface-fuel 'truth' grid."""
    rng = np.random.default_rng(seed)
    comp = _component_fields(nx, ny, rng)

    bd = np.zeros((nz, ny, nx), dtype=np.float32)
    sav_num = np.zeros((nz, ny, nx), dtype=np.float32)
    live_num = np.zeros((nz, ny, nx), dtype=np.float32)

    for name, sav, live in (
        ("litter", SAV_LITTER, LIVE_FRAC_LITTER),
        ("grass", SAV_GRASS, LIVE_FRAC_GRASS),
        ("shrub", SAV_SHRUB, LIVE_FRAC_SHRUB),
    ):
        load2d, h2d = comp[name]
        contrib = _distribute(load2d, h2d, nz)
        bd += contrib
        sav_num += contrib * sav
        live_num += contrib * live

    with np.errstate(invalid="ignore", divide="ignore"):
        sav = np.where(bd > 0, sav_num / bd, 0.0).astype(np.float32)
        live = np.where(bd > 0, live_num / bd, 0.0).astype(np.float32)

    grid = FuelVoxelGrid(
        bulk_density=bd,
        georef=GeoRef(crs=crs, x0=x0, y0=y0, resolution=1.0),
        extra={"sav": sav, "live_fraction": live},
        attrs={
            "scenario": "truth",
            "ecosystem": "longleaf pine / wiregrass (synthetic)",
            "source": "synthetic generator v0.1",
        },
    )
    return grid


def make_fastfuels_baseline(truth: FuelVoxelGrid) -> FuelVoxelGrid:
    """The categorical-fuel-model baseline: domain-uniform vertical profile.

    This is what a single SB40 / FCCS fuel model class yields when painted over
    a landscape — every column is identical, so *all* horizontal heterogeneity
    is lost (CV -> 0). It conserves total mass but misplaces it.
    """
    profile = truth.vertical_profile()  # (nz,)
    bd = np.broadcast_to(profile[:, None, None], truth.shape).astype(np.float32).copy()
    return FuelVoxelGrid(
        bulk_density=bd,
        dz=truth.dz, dy=truth.dy, dx=truth.dx,
        georef=truth.georef,
        extra={
            "sav": np.full(truth.shape, SAV_GRASS, np.float32),
            "live_fraction": np.full(truth.shape, LIVE_FRAC_GRASS, np.float32),
        },
        attrs={"scenario": "fastfuels_baseline",
               "note": "uniform SB40-style fuel model applied to whole domain"},
    )


def make_our_prediction(truth: FuelVoxelGrid, seed: int = 7,
                        sensor_noise: float = 0.15, footprint: float = 1.2,
                        shape_noise: float = 0.04) -> FuelVoxelGrid:
    """Recover heterogeneity from a simulated remote-sensing retrieval.

    This emulates the real two-part LiDAR/SAR method:

    1. **Vertical structure** is *measured* — LiDAR return profiles resolve the
       shape of the fuel column well. We observe each column's normalized
       vertical shape with modest noise.
    2. **Magnitude (fuel load)** is the hard part — it needs calibration. We
       observe a footprint-blurred, noisy load proxy (LiDAR voxel occupancy /
       SAR backscatter) and apply a simple linear calibration (gain 1 here).

    The recovered grid = measured vertical shape x calibrated load, so it keeps
    the sub-metre clumping that ``fastfuels`` discards, with realistic error.
    """
    rng = np.random.default_rng(seed)
    truth_load = truth.fuel_load()

    # (1) Calibrated magnitude: blurred + noisy load proxy.
    obs_load = gaussian_filter(truth_load, sigma=footprint, mode="reflect")
    obs_load = obs_load + rng.normal(0, sensor_noise * truth_load.mean(), obs_load.shape)
    obs_load = np.clip(obs_load, 0, None).astype(np.float32)

    # (2) Measured vertical shape (normalized per column) + noise, renormalized.
    col_sum = truth.bulk_density.sum(axis=0)  # (ny, nx)
    shape = np.divide(truth.bulk_density, col_sum, where=col_sum > 0,
                      out=np.zeros_like(truth.bulk_density))
    shape = np.clip(shape + rng.normal(0, shape_noise, shape.shape), 0, None)
    # cells with no observed structure default to all mass in the bottom voxel
    empty = shape.sum(axis=0) <= 0
    shape[0][empty] = 1.0
    shape = shape / shape.sum(axis=0, keepdims=True)

    # bd_z = shape_z * load / dz  ->  sum_z(bd_z) * dz == load
    bd = (shape * (obs_load / truth.dz)[None, :, :]).astype(np.float32)
    # min-density floor: real voxelizations drop negligible returns so phantom
    # slivers don't masquerade as fuelbed height.
    bd[bd < 0.03] = 0.0

    return FuelVoxelGrid(
        bulk_density=bd,
        dz=truth.dz, dy=truth.dy, dx=truth.dx,
        georef=truth.georef,
        extra={
            "sav": np.full(truth.shape, SAV_GRASS, np.float32),
            "live_fraction": np.full(truth.shape, LIVE_FRAC_GRASS, np.float32),
        },
        attrs={"scenario": "ours",
               "method": "remote-sensing texture -> calibrated load -> voxelize"},
    )


def make_demo_scenario(nx: int = 100, ny: int = 100, nz: int = 4,
                       seed: int = 42) -> DemoScene:
    """Build the full truth / fastfuels / ours comparison scene."""
    truth = make_truth(nx=nx, ny=ny, nz=nz, seed=seed)
    fastfuels = make_fastfuels_baseline(truth)
    ours = make_our_prediction(truth, seed=seed + 1)
    return DemoScene(
        truth=truth,
        fastfuels=fastfuels,
        ours=ours,
        truth_load=truth.fuel_load(),
        fastfuels_load=fastfuels.fuel_load(),
        ours_load=ours.fuel_load(),
    )


# ── Week 2: SAR + LiDAR fusion scenario ──────────────────────────────────────
# To test the fusion method we need a scene with (a) true surface load, (b) an
# overstory CANOPY that occludes airborne LiDAR, and two sensor observations:
#   * LiDAR understory signal — accurate in the open, but attenuated and noisier
#     under canopy (returns thin out where the overstory intercepts pulses);
#   * SAR backscatter — canopy-PENETRATING (nearly canopy-independent) but coarse
#     and noisy with gentle saturation.
# A canopy-weighted blend should then beat either sensor alone, especially under
# canopy — the whole thesis of "radar fills LiDAR's under-canopy blind spot".

@dataclass
class FusionScene:
    truth_load: np.ndarray   # (ny, nx) true surface fuel load (kg/m^2)
    canopy: np.ndarray       # (ny, nx) true overstory cover [0, 1]
    canopy_obs: np.ndarray   # measured canopy cover (LiDAR CHM, low noise)
    lidar_obs: np.ndarray    # LiDAR understory signal (degraded under canopy)
    sar_obs: np.ndarray      # SAR backscatter proxy (canopy-penetrating, coarse)
    uniform_load: np.ndarray # FastFuels/SB40 uniform baseline (domain mean)


def make_canopy_cover(nx, ny, seed=99):
    """Clumpy overstory canopy-cover field in [0, 1] (tree patches)."""
    rng = np.random.default_rng(seed)
    g = _gaussian_field((ny, nx), scale=9.0, rng=rng)
    return np.clip((g - 0.45) / 0.45, 0, 1).astype(np.float32)


def simulate_lidar_obs(L, C, seed=11, occlusion=0.7, base_noise=0.10):
    """Airborne-LiDAR understory observation: accurate in the open, attenuated
    and noisier under canopy (occlusion thins near-ground returns)."""
    rng = np.random.default_rng(seed)
    m = float(L.mean())
    atten = 1.0 - occlusion * C
    noise = rng.normal(0, base_noise * m * (1 + 4 * C), L.shape)
    return np.clip(L * atten + noise, 0, None).astype(np.float32)


def simulate_sar_obs(L, seed=12, footprint=2.5, noise=0.16, sat=1.4):
    """SAR backscatter proxy: penetrates canopy (canopy-independent) but coarse
    (footprint blur), noisy, and saturating at higher loads."""
    rng = np.random.default_rng(seed)
    m = float(L.mean())
    blurred = gaussian_filter(L, sigma=footprint, mode="reflect")
    saturated = sat * (1.0 - np.exp(-blurred / sat))
    return np.clip(saturated + rng.normal(0, noise * m, L.shape), 0, None).astype(np.float32)


@dataclass
class Stage1Scene:
    target: np.ndarray            # (ny, nx) 30 m surface fuel load (kg/m²)
    predictors: dict             # name -> (ny, nx) predictor at 30 m
    canopy: np.ndarray           # (ny, nx) overstory cover [0, 1] (stratum split)
    block_id: np.ndarray         # (ny, nx) spatial-block ids for blocked CV


def make_stage1_scene(n=72, seed=3, blocks=6):
    """Synthetic 30 m region for the Stage-1 surface-fuel regressor.

    A latent vegetation/moisture field drives the true 30 m fuel load. Free
    spaceborne predictors are *partial, noisy views* of that latent: an
    AlphaEarth-like **embedding** (richest), Sentinel-1, Sentinel-2, and terrain.
    Predictor skill DROPS under canopy (occlusion of the surface signal), so
    per-stratum (open vs under-canopy) metrics are honest — and the embedding
    ablation shows the foundation features carry the most signal.
    """
    rng = np.random.default_rng(seed)
    veg = _gaussian_field((n, n), scale=8.0, rng=rng)
    moisture = _gaussian_field((n, n), scale=12.0, rng=rng)
    canopy = np.clip((_gaussian_field((n, n), 9.0, rng) - 0.55) / 0.35, 0, 1).astype(np.float32)

    target = np.clip(0.3 + veg + 0.6 * moisture, 0.05, None).astype(np.float32)

    def noise(s):
        return rng.normal(0, s, (n, n))
    # AlphaEarth-like embedding: rich, low-noise, sees both drivers (best single block);
    # physical sensors are noisier partial views (S1~moisture, S2~veg canopy-attenuated).
    predictors = {
        "embed": (0.9 * veg + 0.8 * moisture - 0.3 * veg * canopy + noise(0.05)).astype(np.float32),
        "s1": (0.6 * moisture + 0.3 * veg + noise(0.13)).astype(np.float32),
        "s2": (0.8 * veg * (1 - 0.3 * canopy) + noise(0.12)).astype(np.float32),
        "terrain": (0.2 * moisture + noise(0.25)).astype(np.float32),
    }
    by = (np.arange(n) * blocks // n)
    block_id = (by[:, None] * blocks + by[None, :]).astype(int)
    return Stage1Scene(target=target, predictors=predictors, canopy=canopy, block_id=block_id)


@dataclass
class DownscaleScene:
    truth: np.ndarray    # (n, n) 1 m surface-fuel load (the ALS-derived target)
    factor: int          # coarsening factor (1 m -> ~30 m baseline)
    s1: np.ndarray       # ~10 m SAR covariate (canopy-penetrating, global)
    embed: np.ndarray    # ~10 m multi-sensor / embedding covariate (S2 + AlphaEarth/Clay)
    canopy: np.ndarray   # canopy cover (global product)


def make_downscale_scene(n=240, seed=7, factor=30):
    """Multi-scale surface-fuel scene for the 30 m -> 1 m downscaler.

    Real surface fuels vary at *several* scales: landscape/community structure
    (~10-40 m, recoverable from 10 m covariates) plus sub-10 m clumps (the
    information-limited band). Globally-available 10 m covariates (Sentinel-1,
    Sentinel-2, AlphaEarth/Clay embeddings) are simulated as blurred + noisy
    observations that resolve the medium band but not the finest detail.
    """
    rng = np.random.default_rng(seed)
    medium = 0.55 * _gaussian_field((n, n), scale=18.0, rng=rng)  # 10-40 m communities
    fine = 0.55 * _gaussian_field((n, n), scale=2.5, rng=rng)     # sub-10 m clumps
    truth = np.clip(0.30 + medium + fine, 0, None).astype(np.float32)
    m = float(truth.mean())
    s1 = np.clip(gaussian_filter(truth, 6.0, mode="reflect")
                 + rng.normal(0, 0.12 * m, truth.shape), 0, None).astype(np.float32)
    embed = np.clip(gaussian_filter(truth, 4.0, mode="reflect")
                    + rng.normal(0, 0.07 * m, truth.shape), 0, None).astype(np.float32)
    canopy = make_canopy_cover(n, n, seed=seed + 57)
    return DownscaleScene(truth=truth, factor=factor, s1=s1, embed=embed, canopy=canopy)


def make_fusion_scenario(nx=120, ny=120, seed=42):
    """Build the SAR+LiDAR fusion test scene."""
    truth = make_truth(nx=nx, ny=ny, nz=4, seed=seed)
    L = truth.fuel_load()
    C = make_canopy_cover(nx, ny, seed=seed + 57)
    rng = np.random.default_rng(seed + 3)
    return FusionScene(
        truth_load=L,
        canopy=C,
        canopy_obs=np.clip(C + rng.normal(0, 0.05, C.shape), 0, 1).astype(np.float32),
        lidar_obs=simulate_lidar_obs(L, C, seed=seed + 11),
        sar_obs=simulate_sar_obs(L, seed=seed + 12),
        uniform_load=np.full_like(L, float(L.mean())),
    )
