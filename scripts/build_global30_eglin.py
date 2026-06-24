"""Real Stage-1 run over the Eglin 3DEP tile: predict a 30 m surface-fuel target
from free spaceborne predictors (AlphaEarth embeddings + Sentinel-1 + terrain),
validated with spatially-blocked CV. First end-to-end Stage-1 on real data.

Target (proxy, pending FIA/NEON calibration — the documented biggest risk):
near-ground (0.15-4 m) 3DEP return density per 30 m cell, scaled to a literature
mean load (0.6 kg/m²). Predictors are all global/free.

Scope: ONE ~1.5 km tile (~50x50 cells, single ecosystem) — a real POC, AOI-based;
scaling to an ecoregion + adding Sentinel-2/GEDI + FIA/NEON validation are next.

Usage:  python scripts/build_global30_eglin.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, GeoRef, lidar, global30 as g30, embeddings as emb, sar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAZ = os.path.join(ROOT, "data", "raw", "eglin_3dep_000051.laz")
RES = 30.0


def build_target_grid(pc):
    """30 m grid over the whole tile: fuel proxy, canopy cover, terrain, blocks."""
    x0, y0 = float(pc.x.min()), float(pc.y.min())
    nx = int((pc.x.max() - x0) // RES)
    ny = int((pc.y.max() - y0) // RES)
    grid = FuelVoxelGrid(bulk_density=np.zeros((1, ny, nx), np.float32),
                         dz=RES, dy=RES, dx=RES,
                         georef=GeoRef("EPSG:32616", x0, y0, RES))
    # height above ground
    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, max(nx, ny) * RES, res=5.0)
    gi = np.clip(((pc.x - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((pc.y - y0) / dres).astype(int), 0, ng - 1)
    h = pc.z - dtm[gj, gi]
    ix = np.clip(((pc.x - x0) / RES).astype(int), 0, nx - 1)
    iy = np.clip(((pc.y - y0) / RES).astype(int), 0, ny - 1)
    # near-ground (understory/surface) return density -> fuel proxy
    surf = (h >= 0.15) & (h < 4.0)
    counts = np.zeros((ny, nx), np.float32)
    np.add.at(counts, (iy[surf], ix[surf]), 1.0)
    target = counts * (0.6 / max(counts.mean(), 1e-6))           # scale to ~0.6 kg/m²
    canopy = lidar.canopy_cover_grid(pc, grid)                    # fraction returns >2 m
    # terrain
    elev = np.zeros((ny, nx), np.float32); tot = np.zeros((ny, nx), np.float32)
    g = pc.cls == 2
    np.add.at(elev, (iy[g], ix[g]), pc.z[g]); np.add.at(tot, (iy[g], ix[g]), 1.0)
    elev = np.divide(elev, tot, out=np.zeros_like(elev), where=tot > 0)
    gy, gx = np.gradient(elev); slope = np.hypot(gy, gx)
    b = 3
    by = (np.arange(ny) * b // ny); bx = (np.arange(nx) * b // nx)
    block_id = by[:, None] * b + bx[None, :]
    return grid, target.astype(np.float32), canopy, (elev - elev.mean()).astype(np.float32), slope.astype(np.float32), block_id


def main():
    if not os.path.exists(LAZ):
        sys.exit("Missing LAZ — run scripts/download_eglin_lidar.py")
    print("Loading 3DEP point cloud...")
    pc = lidar.load_points(LAZ, src_epsg=2238, target_epsg=32616, z_unit="ft_us")
    grid, target, canopy, elev, slope, block_id = build_target_grid(pc)
    print(f"30 m grid {target.shape} over {grid.nx*RES/1000:.1f}x{grid.ny*RES/1000:.1f} km; "
          f"target mean {target.mean():.2f} kg/m²; under-canopy {(canopy>=0.3).mean()*100:.0f}%")

    print("Fetching AlphaEarth embeddings (free, Source Coop mirror)...")
    aef = emb.aef_for_grid(grid, year=2024)
    print("Fetching Sentinel-1 seasonal (Planetary Computer)...")
    s1 = sar.sar_features_for_grid(grid)

    preds = {"canopy": canopy, "elev": elev, "slope": slope}
    if s1 is not None:
        preds["s1_vhvv"], preds["s1_rvi"] = s1["vh_vv"], s1["rvi"]
    aef_keys = []
    if aef is not None:
        from sklearn.decomposition import PCA
        flat = np.nan_to_num(aef.reshape(64, -1).T)
        k = 16
        pcs = PCA(n_components=k, random_state=0).fit_transform(flat).T.reshape(k, grid.ny, grid.nx)
        for i in range(k):
            preds[f"aef{i:02d}"] = pcs[i].astype(np.float32); aef_keys.append(f"aef{i:02d}")
        print(f"  AlphaEarth -> top {k} PCA components")

    def run(p, label):
        out = g30.blocked_cv(target, p, block_id, conformal=True)
        rep = g30.evaluate(out, target, canopy)
        print(f"  {label:<34} R2 {rep['r2']:>6}  open {rep['r2_open']:>6}  canopy {rep['r2_under_canopy']:>6}  cov {rep['interval_coverage']:>5}")
        return rep

    print("\nSpatially-blocked CV (real 30 m Eglin, vs 3DEP fuel proxy):")
    run(preds, "full (AEF + S1 + terrain + canopy)")
    run({k: v for k, v in preds.items() if k not in aef_keys}, "physical only (S1 + terrain + canopy)")
    if aef_keys:
        run({k: preds[k] for k in aef_keys}, "AlphaEarth only")

    np.savez_compressed(os.path.join(ROOT, "data", "processed", "global30_eglin.npz"),
                        target=target, canopy=canopy, aef_available=aef is not None)
    print("\nNote: target is a 3DEP near-ground return-density PROXY (pending FIA/NEON calibration). "
          "Single-tile POC; scale to ecoregion + add S2/GEDI next.")


if __name__ == "__main__":
    main()
