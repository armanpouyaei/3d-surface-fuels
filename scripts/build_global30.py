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

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import FuelVoxelGrid, GeoRef, lidar, global30 as g30, embeddings as emb, sar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = 30.0
TARGET_EPSG = 32616  # set per-run in main()


def build_target_grid(pc):
    """30 m grid over the whole tile: fuel proxy, canopy cover, terrain, blocks."""
    x0, y0 = float(pc.x.min()), float(pc.y.min())
    nx = int((pc.x.max() - x0) // RES)
    ny = int((pc.y.max() - y0) // RES)
    grid = FuelVoxelGrid(bulk_density=np.zeros((1, ny, nx), np.float32),
                         dz=RES, dy=RES, dx=RES,
                         georef=GeoRef(f"EPSG:{TARGET_EPSG}", x0, y0, RES))
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
    global TARGET_EPSG
    ap = argparse.ArgumentParser()
    ap.add_argument("--laz", default=os.path.join(ROOT, "data", "raw", "eglin_3dep_000051.laz"))
    ap.add_argument("--src-epsg", type=int, default=2238, help="LAZ horizontal CRS (Eglin 2238, OSBS 6438)")
    ap.add_argument("--target-epsg", type=int, default=32616, help="UTM target (Eglin 32616, OSBS 32617)")
    ap.add_argument("--aef-year", type=int, default=2024, help="AlphaEarth year (match the LiDAR vintage)")
    ap.add_argument("--site", default="eglin")
    args = ap.parse_args()
    TARGET_EPSG = args.target_epsg

    if not os.path.exists(args.laz):
        sys.exit(f"Missing LAZ: {args.laz}")
    print(f"[{args.site}] Loading 3DEP point cloud (EPSG:{args.src_epsg} -> UTM {args.target_epsg})...")
    pc = lidar.load_points(args.laz, src_epsg=args.src_epsg, target_epsg=args.target_epsg, z_unit="ft_us")
    grid, target, canopy, elev, slope, block_id = build_target_grid(pc)
    print(f"30 m grid {target.shape} over {grid.nx*RES/1000:.1f}x{grid.ny*RES/1000:.1f} km; "
          f"target mean {target.mean():.2f} kg/m²; under-canopy {(canopy>=0.3).mean()*100:.0f}%")

    print(f"Fetching AlphaEarth embeddings (year {args.aef_year}, free Source Coop mirror)...")
    aef = emb.aef_for_grid(grid, year=args.aef_year)
    print("Fetching Sentinel-1 seasonal (Planetary Computer)...")
    s1 = sar.sar_features_for_grid(grid)

    # Predictors must be SPACEBORNE-only (the product is applied where there is no airborne
    # LiDAR). LiDAR-derived canopy would leak the target (same sensor) — keep it ONLY for
    # per-stratum metrics, not as a feature. Terrain (static ground elevation) is fine — it is
    # independent of the fuel target and globally available (Copernicus DEM equivalent).
    preds = {"elev": elev, "slope": slope}
    if s1 is not None:
        preds["s1_vhvv"], preds["s1_rvi"] = s1["vh_vv"], s1["rvi"]
    aef_keys = []
    if aef is not None:
        deq = np.nan_to_num(aef)  # all 64 dequantized bands (fairest test of AEF)
        for i in range(64):
            preds[f"aef{i:02d}"] = deq[i].astype(np.float32); aef_keys.append(f"aef{i:02d}")
        print("  AlphaEarth -> all 64 bands")

    def run(p):
        out = g30.blocked_cv(target, p, block_id, conformal=True)
        return out, g30.evaluate(out, target, canopy)

    print(f"\nSpatially-blocked CV (real 30 m {args.site}, vs 3DEP fuel proxy):")
    sets = {"full": preds, "physical": {k: v for k, v in preds.items() if k not in aef_keys}}
    if aef_keys:
        sets["AEF-only"] = {k: preds[k] for k in aef_keys}
    outs, reps = {}, {}
    for label, p in sets.items():
        outs[label], reps[label] = run(p)
        r = reps[label]
        print(f"  {label:<12} R2 {r['r2']:>6}  open {r['r2_open']:>6}  canopy {r['r2_under_canopy']:>6}  cov {r['interval_coverage']:>5}")

    # ---- figure: ablation bars + predicted-vs-observed (full) ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    methods = list(reps); strata = [("overall", "r2"), ("open", "r2_open"), ("under-canopy", "r2_under_canopy")]
    xs = np.arange(len(methods)); w = 0.26
    for i, (sl, key) in enumerate(strata):
        ax[0].bar(xs + (i - 1) * w, [reps[m][key] for m in methods], w, label=sl)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(methods); ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_ylabel("R²"); ax[0].set_title(f"{args.site.upper()} Stage-1 ablation (blocked CV)"); ax[0].legend(fontsize=8)
    t, p = target.ravel(), outs["full"]["median"].ravel()
    ax[1].scatter(t, p, s=10, alpha=.4, color="#2c7fb8"); lim = [0, max(t.max(), p.max())]
    ax[1].plot(lim, lim, "r--", lw=1); ax[1].set_xlabel("3DEP fuel proxy (target, kg/m²)")
    ax[1].set_ylabel("predicted (kg/m²)"); ax[1].set_title(f"full model — R²={reps['full']['r2']}, coverage={reps['full']['interval_coverage']}")
    plt.tight_layout(); fig_path = os.path.join(ROOT, "figures", f"global30_{args.site}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True); plt.savefig(fig_path, dpi=110)
    print(f"  saved {fig_path}")

    np.savez_compressed(os.path.join(ROOT, "data", "processed", f"global30_{args.site}.npz"),
                        target=target, canopy=canopy, aef_available=aef is not None)
    print("\nNote: target is a 3DEP near-ground return-density PROXY (pending FIA/NEON calibration). "
          "Single-tile POC; scale to ecoregion + add S2/GEDI next.")


if __name__ == "__main__":
    main()
