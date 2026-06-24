"""Real airborne-LiDAR -> 1 m surface-fuel voxel grid.

Implements the simple, explainable core of "Approach A": take a LiDAR point
cloud, height-normalize it to a ground surface, keep the near-ground
(0-`nz` m) stratum, voxelize returns into 1 m bins, and convert per-voxel
return density to bulk density via a calibration.

The pattern (where fuel is, how it stacks vertically) is *measured*; the
absolute magnitude is anchored to a literature mean load until co-located
destructive/TLS truth is available for a proper local calibration — exactly the
"measured heterogeneity, calibrated magnitude" design in IDEAS.md.

Tested on USGS 3DEP tile FL_OKALOOSACO_2007 over Eglin AFB (EPSG:2238, ftUS,
~4.3 pts/m^2). Returns a `FuelVoxelGrid`, so metrics/dashboard work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .voxel import FuelVoxelGrid, GeoRef

US_FOOT = 0.3048006096  # US survey foot -> metre


@dataclass
class PointCloud:
    x: np.ndarray  # metres, target CRS (UTM 16N)
    y: np.ndarray
    z: np.ndarray  # metres
    cls: np.ndarray
    epsg: int


def load_points(path: str, src_epsg: int = 2238, target_epsg: int = 32616,
                z_unit: str = "ft_us") -> PointCloud:
    """Read a LAZ/LAS file and return points in metres in ``target_epsg``.

    ``src_epsg`` is the file's CRS (3DEP FL_OKALOOSACO_2007 = 2238, ftUS) used
    when the file's own CRS VLR does not parse. Horizontal coords are reprojected
    to ``target_epsg`` (UTM 16N); vertical is converted to metres per ``z_unit``.
    """
    import laspy
    from pyproj import Transformer

    f = laspy.read(path)
    x = np.asarray(f.x, dtype=np.float64)
    y = np.asarray(f.y, dtype=np.float64)
    z = np.asarray(f.z, dtype=np.float64)
    cls = np.asarray(f.classification, dtype=np.uint8)

    if z_unit in ("ft_us", "ft"):
        z = z * (US_FOOT if z_unit == "ft_us" else 0.3048)

    if src_epsg != target_epsg:
        tr = Transformer.from_crs(src_epsg, target_epsg, always_xy=True)
        x, y = tr.transform(x, y)

    return PointCloud(x=x, y=y, z=z, cls=cls, epsg=target_epsg)


def _cover_map(pc: PointCloud, res=30.0, height_thresh=2.0):
    """Coarse canopy-cover map (fraction of returns > height_thresh) over the tile."""
    x0, y0 = pc.x.min(), pc.y.min()
    ngx = int(np.ceil((pc.x.max() - x0) / res))
    ngy = int(np.ceil((pc.y.max() - y0) / res))
    g = pc.cls == 2
    gi = np.clip(((pc.x[g] - x0) / res).astype(int), 0, ngx - 1)
    gj = np.clip(((pc.y[g] - y0) / res).astype(int), 0, ngy - 1)
    ground = np.full((ngy, ngx), np.inf)
    np.minimum.at(ground, (gj, gi), pc.z[g])
    if np.isfinite(ground).any():
        ground[~np.isfinite(ground)] = np.nanmin(ground[np.isfinite(ground)])
    else:
        ground[:] = 0.0
    ai = np.clip(((pc.x - x0) / res).astype(int), 0, ngx - 1)
    aj = np.clip(((pc.y - y0) / res).astype(int), 0, ngy - 1)
    tall = (pc.z - ground[aj, ai]) > height_thresh
    tot = np.zeros((ngy, ngx)); np.add.at(tot, (aj, ai), 1.0)
    tc = np.zeros((ngy, ngx)); np.add.at(tc, (aj, ai), tall.astype(float))
    cover = np.divide(tc, tot, out=np.zeros_like(tc), where=tot > 0)
    return cover, x0, y0, res


def find_aoi(pc: PointCloud, size=240.0, cover_max=0.22,
             res=30.0, height_thresh=2.0):
    """Pick an open-but-structured AOI: low mean canopy cover (few trees) yet
    with spatial structure (scattered clumps) so features are identifiable in
    both the LiDAR fuel map and the satellite imagery — making same-spot
    alignment visible. Returns ((cx, cy), info)."""
    cover, x0, y0, res = _cover_map(pc, res, height_thresh)
    ngy, ngx = cover.shape
    win = max(1, int(round(size / res)))
    best = None  # (structure, i, j, mean_cover)
    for j in range(0, max(1, ngy - win)):
        for i in range(0, max(1, ngx - win)):
            blk = cover[j:j + win, i:i + win]
            mc = float(blk.mean())
            if mc <= cover_max:
                sc = float(blk.std())
                if best is None or sc > best[0]:
                    best = (sc, i, j, mc)
    if best is None:  # fall back to least-treed window
        j, i = np.unravel_index(np.argmin(
            np.array([[cover[a:a + win, b:b + win].mean()
                       for b in range(max(1, ngx - win))]
                      for a in range(max(1, ngy - win))])), (ngy - win, ngx - win))
        best = (0.0, int(i), int(j), float(cover[j:j + win, i:i + win].mean()))
    sc, i, j, mc = best
    cx = x0 + (i + win / 2.0) * res
    cy = y0 + (j + win / 2.0) * res
    return (cx, cy), {"mean_cover": round(mc, 3), "structure": round(sc, 3)}


def _ground_dtm(pc: PointCloud, x0, y0, size, res=5.0):
    """Coarse ground surface (min-Z of ground returns) over the AOI, gap-filled."""
    from scipy.ndimage import distance_transform_edt

    ng = int(np.ceil(size / res))
    ground = pc.cls == 2
    gx, gy, gz = pc.x[ground], pc.y[ground], pc.z[ground]
    in_aoi = (gx >= x0) & (gx < x0 + size) & (gy >= y0) & (gy < y0 + size)
    gx, gy, gz = gx[in_aoi], gy[in_aoi], gz[in_aoi]

    dtm = np.full((ng, ng), np.inf)
    if len(gz):
        gi = np.clip(((gx - x0) / res).astype(int), 0, ng - 1)
        gj = np.clip(((gy - y0) / res).astype(int), 0, ng - 1)
        np.minimum.at(dtm, (gj, gi), gz)
    empty = ~np.isfinite(dtm)
    if empty.any():  # nearest-neighbour gap fill
        idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
        filled = dtm[tuple(idx)]
        dtm = np.where(empty, filled, dtm)
    if not np.isfinite(dtm).all():
        dtm[~np.isfinite(dtm)] = np.nanmin(dtm[np.isfinite(dtm)]) if np.isfinite(dtm).any() else 0.0
    return dtm, res, ng


def voxelize(pc: PointCloud, center: Optional[Tuple[float, float]] = None,
             size: float = 240.0, nz: int = 4, dz: float = 1.0,
             target_load_kg_m2: float = 0.6,
             dtm_res: float = 5.0) -> FuelVoxelGrid:
    """Voxelize the near-ground stratum into a 1 m surface-fuel bulk-density grid.

    Parameters
    ----------
    center : (x, y) in target CRS metres; defaults to the cloud's centroid.
    size : AOI side length (m).
    nz, dz : vertical voxels and their height (m) — the 0..nz*dz surface stratum.
    target_load_kg_m2 : domain-mean surface load used to scale return density to
        bulk density (longleaf surface ~0.3-1.0 kg/m^2; default 0.6). The spatial
        pattern is measured; only the overall magnitude is anchored here.
    """
    if center is None:
        center = (float(pc.x.mean()), float(pc.y.mean()))
    x0, y0 = center[0] - size / 2.0, center[1] - size / 2.0
    nx = ny = int(round(size / dz))

    dtm, dres, ng = _ground_dtm(pc, x0, y0, size, res=dtm_res)

    in_aoi = (pc.x >= x0) & (pc.x < x0 + size) & (pc.y >= y0) & (pc.y < y0 + size)
    px, py, pz = pc.x[in_aoi], pc.y[in_aoi], pc.z[in_aoi]

    # height above ground via the coarse DTM
    gi = np.clip(((px - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((py - y0) / dres).astype(int), 0, ng - 1)
    h = pz - dtm[gj, gi]

    surface = (h >= 0) & (h < nz * dz)
    sx, sy, sh = px[surface], py[surface], h[surface]

    # 1 m voxel return counts -> (z, y, x)
    ix = np.clip(((sx - x0) / dz).astype(int), 0, nx - 1)
    iy = np.clip(((sy - y0) / dz).astype(int), 0, ny - 1)
    iz = np.clip((sh / dz).astype(int), 0, nz - 1)
    counts = np.zeros((nz, ny, nx), dtype=np.float32)
    np.add.at(counts, (iz, iy, ix), 1.0)

    # calibrate return density -> bulk density so the domain-mean load matches the anchor
    raw_load = counts.sum(axis=0)            # returns per column
    mean_raw = float(raw_load.mean())
    scale = (target_load_kg_m2 / mean_raw) if mean_raw > 0 else 0.0
    bulk_density = (counts * scale).astype(np.float32)  # kg/m^3 (dz=1 -> load=sum)

    return FuelVoxelGrid(
        bulk_density=bulk_density,
        dz=dz, dy=dz, dx=dz,
        georef=GeoRef(crs=f"EPSG:{pc.epsg}", x0=x0, y0=y0, resolution=dz),
        extra={"return_density": counts},
        attrs={
            "scenario": "measured",
            "source": "USGS 3DEP LiDAR FL_OKALOOSACO_2007 (Eglin AFB)",
            "ecosystem": "longleaf pine sandhill (real)",
            "calibration": f"return-density scaled to mean load {target_load_kg_m2} kg/m2 (literature anchor)",
            "note": "spatial/vertical pattern measured; magnitude pending local destructive calibration",
        },
    )


def canopy_cover_grid(pc: PointCloud, grid: FuelVoxelGrid,
                      height_thresh: float = 2.0) -> np.ndarray:
    """Per-cell canopy cover [0,1] (fraction of returns above ``height_thresh``)
    aligned to ``grid`` (row 0 = south). Used as the SAR/LiDAR fusion weight."""
    x0, y0 = grid.georef.x0, grid.georef.y0
    size = grid.nx * grid.dx
    ny, nx, dz = grid.ny, grid.nx, grid.dx
    dtm, dres, ngc = _ground_dtm(pc, x0, y0, size, res=5.0)
    m = (pc.x >= x0) & (pc.x < x0 + size) & (pc.y >= y0) & (pc.y < y0 + size)
    px, py, pz = pc.x[m], pc.y[m], pc.z[m]
    gi = np.clip(((px - x0) / dres).astype(int), 0, ngc - 1)
    gj = np.clip(((py - y0) / dres).astype(int), 0, ngc - 1)
    h = pz - dtm[gj, gi]
    ci = np.clip(((px - x0) / dz).astype(int), 0, nx - 1)
    cj = np.clip(((py - y0) / dz).astype(int), 0, ny - 1)
    tot = np.zeros((ny, nx)); np.add.at(tot, (cj, ci), 1.0)
    tall = np.zeros((ny, nx)); np.add.at(tall, (cj, ci), (h > height_thresh).astype(float))
    cover = np.divide(tall, tot, out=np.zeros_like(tot), where=tot > 0)
    return cover.astype(np.float32)


def uniform_baseline(grid: FuelVoxelGrid) -> FuelVoxelGrid:
    """FastFuels/LANDFIRE-style uniform surface layer for the same AOI.

    Sets every column to the domain-mean vertical profile — what a single SB40
    fuel-model class produces (CV -> 0). Lets the dashboard show measured-vs-
    uniform on real data even before LANDFIRE FBFM40 is wired in.
    """
    profile = grid.vertical_profile()
    bd = np.broadcast_to(profile[:, None, None], grid.shape).astype(np.float32).copy()
    return FuelVoxelGrid(
        bulk_density=bd, dz=grid.dz, dy=grid.dy, dx=grid.dx, georef=grid.georef,
        attrs={"scenario": "uniform_baseline",
               "note": "domain-mean profile (FastFuels/SB40-style uniform layer)"},
    )
