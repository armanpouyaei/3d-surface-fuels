"""END-TO-END measured-structure pipeline over OSBS (the headline deliverable).

Proves the thesis in one honest chain, against real measured truth:

  Stage-1 : spaceborne (AlphaEarth + Sentinel-1 + terrain) -> 30 m fuel STRUCTURE
            (near-ground bulk-density proxy), spatially-blocked CV  -> out-of-fold 30 m map
  Stage-2 : downscale the *predicted* 30 m map -> fine (10 m) using AlphaEarth 10 m
            covariates + terrain, mass-conserving, spatially held-out
  Validate: against the MEASURED 3DEP structure (truth), head-to-head vs the
            FastFuels-style UNIFORM surface layer (one value per AOI/class).

Honest error budget reported:
  * Stage-1 30 m accuracy (spaceborne skill at the coarse scale),
  * Stage-2 *within-block* skill (the sub-30 m detail a 30 m product cannot have),
  * end-to-end fine accuracy (Stage-1 error propagated + Stage-2 detail),
  * vs FastFuels uniform (within-block R2 = 0 by construction — the gap we fill).

Where airborne LiDAR exists we deliver the measured 1 m structure directly (the
truth panel); this pipeline is how the SAME product generalizes where only
spaceborne data exist.

Usage:  python scripts/build_pipeline_osbs.py
        (needs data/raw/osbs_3dep_2018.laz; AEF index is cached)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import (FuelVoxelGrid, GeoRef, lidar, metrics as M,  # noqa: E402
                           global30 as g30, downscale as ds, embeddings as emb, sar)

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAZ = os.path.join(ROOT, "data", "raw", "osbs_3dep_2018.laz")
SRC_EPSG, TGT_EPSG, AEF_YEAR, SITE = 6438, 32617, 2018, "osbs"
RES30, FACTOR = 30.0, 3          # fine = 10 m (AlphaEarth native); 3x3 cells per 30 m block
RES10 = RES30 / FACTOR


# ── measured structure (near-ground 0.15-4 m return fraction -> bulk-density proxy) ──
def grid_structure(pc, h, x0, y0, nx, ny, res):
    ix = np.clip(((pc.x - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((pc.y - y0) / res).astype(int), 0, ny - 1)
    surf = (h >= 0.15) & (h < 4.0)
    near = np.zeros((ny, nx), np.float32); np.add.at(near, (iy[surf], ix[surf]), 1.0)
    tot = np.zeros((ny, nx), np.float32); np.add.at(tot, (iy, ix), 1.0)
    frac = np.divide(near, tot, out=np.zeros_like(near), where=tot > 0)
    return frac, tot


def terrain(pc, x0, y0, nx, ny, res):
    elev = np.zeros((ny, nx), np.float32); tot = np.zeros((ny, nx), np.float32)
    g = pc.cls == 2
    ix = np.clip(((pc.x[g] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((pc.y[g] - y0) / res).astype(int), 0, ny - 1)
    np.add.at(elev, (iy, ix), pc.z[g]); np.add.at(tot, (iy, ix), 1.0)
    elev = np.divide(elev, tot, out=np.zeros_like(elev), where=tot > 0)
    # fill empty cells with global mean so gradients are sane
    elev[tot == 0] = elev[tot > 0].mean() if (tot > 0).any() else 0.0
    gy, gx = np.gradient(elev); slope = np.hypot(gy, gx)
    return (elev - elev.mean()).astype(np.float32), slope.astype(np.float32)


def aef_pca(grid10, k=8):
    aef = emb.aef_for_grid(grid10, year=AEF_YEAR)
    if aef is None:
        return None, None
    flat = np.nan_to_num(aef).reshape(64, -1).T  # (npix, 64), row0=south already
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    z = StandardScaler().fit_transform(flat)
    comps = PCA(n_components=k, random_state=0).fit_transform(z)
    return aef, comps.T.reshape(k, grid10.ny, grid10.nx).astype(np.float32)


# ── honest end-to-end downscale: train on TRUE coarse, predict with PREDICTED coarse ──
def end_to_end_downscale(truth_fine, true_coarse, pred_coarse, covars, factor):
    ny, nx = truth_fine.shape
    cu_true = ds.upsample(true_coarse, factor, truth_fine.shape)
    cu_pred = ds.upsample(pred_coarse, factor, truth_fine.shape)
    Xtr, _ = ds.stack_features(cu_true, covars, factor)   # train features: TRUE 30 m baseline
    Xpr, _ = ds.stack_features(cu_pred, covars, factor)   # deploy features: PREDICTED 30 m baseline
    y = truth_fine.ravel()
    pred = np.zeros_like(y)
    for k in range(4):
        test = ds._quadrant_mask(ny, nx, k).ravel()
        est = ds.make_estimator().fit(Xtr[~test], y[~test])
        pred[test] = est.predict(Xpr[test])           # held-out block, deployed baseline
    pred = np.clip(pred.reshape(ny, nx), 0, None)
    return ds.enforce_mass_conservation(pred, pred_coarse, factor)  # conserve to deployed 30 m


def stats(p, truth, factor):
    bd = lambda a: a - ds.upsample(ds.block_coarsen(a, factor), factor, a.shape)
    return {"r2": round(M.r2(p, truth), 3),
            "within_block_r2": round(M.r2(bd(p), bd(truth)), 3),
            "cv": round(float(p.std() / (p.mean() + 1e-9)), 3)}


def main():
    global LAZ, SRC_EPSG, TGT_EPSG, AEF_YEAR, SITE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--laz", default=LAZ)
    ap.add_argument("--src-epsg", type=int, default=SRC_EPSG)
    ap.add_argument("--target-epsg", type=int, default=TGT_EPSG)
    ap.add_argument("--aef-year", type=int, default=AEF_YEAR)
    ap.add_argument("--z-unit", default="ft_us", help="LAZ vertical unit: ft_us or m")
    a = ap.parse_args()
    SITE, LAZ, SRC_EPSG, TGT_EPSG, AEF_YEAR = a.site, a.laz, a.src_epsg, a.target_epsg, a.aef_year

    if not os.path.exists(LAZ):
        sys.exit(f"Missing {LAZ}")
    print(f"Loading {SITE.upper()} 3DEP {AEF_YEAR} (EPSG:{SRC_EPSG} -> UTM {TGT_EPSG}, z={a.z_unit})...")
    pc = lidar.load_points(LAZ, src_epsg=SRC_EPSG, target_epsg=TGT_EPSG, z_unit=a.z_unit)
    x0, y0 = float(pc.x.min()), float(pc.y.min())
    nx30 = int((pc.x.max() - x0) // RES30); ny30 = int((pc.y.max() - y0) // RES30)
    nx10, ny10 = nx30 * FACTOR, ny30 * FACTOR
    print(f"  {len(pc.x):,} pts; AOI {nx30}x{ny30} @30 m ({nx30*RES30/1000:.1f}x{ny30*RES30/1000:.1f} km)")

    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, max(nx10, ny10) * RES10, res=5.0)
    gi = np.clip(((pc.x - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((pc.y - y0) / dres).astype(int), 0, ng - 1)
    h = pc.z - dtm[gj, gi]

    # measured structure: truth at 10 m, coarsen to 30 m (perfect nesting)
    truth10, _ = grid_structure(pc, h, x0, y0, nx10, ny10, RES10)
    scale = 0.6 / max(truth10[truth10 > 0].mean(), 1e-6)        # -> ~0.6 kg/m2 mean (interpretable)
    truth10 *= scale
    truth30 = ds.block_coarsen(truth10, FACTOR)
    elev30, slope30 = terrain(pc, x0, y0, nx30, ny30, RES30)
    elev10, slope10 = terrain(pc, x0, y0, nx10, ny10, RES10)

    grid30 = FuelVoxelGrid(np.zeros((1, ny30, nx30), np.float32), dz=RES30, dy=RES30, dx=RES30,
                           georef=GeoRef(f"EPSG:{TGT_EPSG}", x0, y0, RES30))
    grid10 = FuelVoxelGrid(np.zeros((1, ny10, nx10), np.float32), dz=RES10, dy=RES10, dx=RES10,
                           georef=GeoRef(f"EPSG:{TGT_EPSG}", x0, y0, RES10))
    canopy30 = lidar.canopy_cover_grid(pc, grid30)

    # ── STAGE 1: spaceborne -> 30 m structure ──────────────────────────────────
    print(f"Stage-1: AlphaEarth({AEF_YEAR}) + Sentinel-1 + terrain -> 30 m structure...")
    aef30 = emb.aef_for_grid(grid30, year=AEF_YEAR)
    s1_30 = sar.sar_features_for_grid(grid30)
    if aef30 is None:
        sys.exit("AEF (30 m) fetch failed")
    preds = {"elev": elev30, "slope": slope30}
    if s1_30 is not None:
        preds["s1_vhvv"], preds["s1_rvi"] = s1_30["vh_vv"], s1_30["rvi"]
    for i in range(64):
        preds[f"aef{i:02d}"] = np.nan_to_num(aef30[i]).astype(np.float32)
    b = 3
    block_id = (np.arange(ny30) * b // ny30)[:, None] * b + (np.arange(nx30) * b // nx30)[None, :]
    out30 = g30.blocked_cv(truth30, preds, block_id, conformal=True)
    pred30 = out30["median"]
    rep30 = g30.evaluate(out30, truth30, canopy30)
    print(f"  Stage-1 30 m: R2 {rep30['r2']}  open {rep30.get('r2_open')}  "
          f"under-canopy {rep30.get('r2_under_canopy')}  interval-cov {rep30['interval_coverage']}")

    # ── STAGE 2: downscale 30 m -> 10 m with AlphaEarth 10 m + terrain ──────────
    print(f"Stage-2: AlphaEarth 10 m (PCA) + terrain -> downscale to {RES10:.0f} m...")
    _, aefpc10 = aef_pca(grid10, k=8)
    if aefpc10 is None:
        sys.exit("AEF (10 m) fetch failed")
    covars = {f"aef_pc{i}": aefpc10[i] for i in range(aefpc10.shape[0])}
    covars["elev"], covars["slope"] = elev10, slope10
    s1_10 = sar.sar_features_for_grid(grid10)
    if s1_10 is not None:
        covars["s1_vhvv"], covars["s1_rvi"] = s1_10["vh_vv"].astype(np.float32), s1_10["rvi"].astype(np.float32)

    # intrinsic downscaler skill (clean coarsening), then end-to-end with predicted coarse
    clean = ds.downscale_cv(truth10, truth30, covars, FACTOR)
    e2e = end_to_end_downscale(truth10, truth30, pred30, covars, FACTOR)

    # ── products vs measured truth (10 m) ──────────────────────────────────────
    uniform = np.full_like(truth10, truth10.mean())            # FastFuels-style uniform layer
    stage1_up = ds.upsample(pred30, FACTOR, truth10.shape)     # 30 m product, naive upsample
    products = {"FastFuels uniform": uniform, "Stage-1 30 m (upsampled)": stage1_up,
                "Downscaler (clean coarse)": clean["downscaled"], "End-to-end (ours)": e2e}
    print(f"\n{'product':<28}{'R2':>8}{'within-block R2':>17}{'CV':>7}  (truth CV {truth10.std()/truth10.mean():.2f})")
    reps = {}
    for name, p in products.items():
        reps[name] = stats(p, truth10, FACTOR)
        print(f"{name:<28}{reps[name]['r2']:>8}{reps[name]['within_block_r2']:>17}{reps[name]['cv']:>7}")

    # ── figure ─────────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.25, 1])
    vmax = float(np.percentile(truth10, 98))
    panels = [("Measured 3DEP (truth)", truth10), ("FastFuels uniform", uniform),
              ("Stage-1 30 m (spaceborne)", stage1_up), ("End-to-end 10 m (ours)", e2e)]
    for j, (title, arr) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(arr, origin="lower", cmap="YlOrBr", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[:4], fraction=0.012, pad=0.01, label="structure (kg/m² proxy)")

    names = list(products)
    axb = fig.add_subplot(gs[1, :2])
    xs = np.arange(len(names)); w = 0.38
    axb.bar(xs - w / 2, [reps[n]["r2"] for n in names], w, label="overall R²", color="#2c7fb8")
    axb.bar(xs + w / 2, [reps[n]["within_block_r2"] for n in names], w, label="within-block R² (sub-30 m)", color="#d95f0e")
    axb.axhline(0, color="k", lw=.6); axb.set_xticks(xs)
    axb.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    axb.set_ylabel("R²"); axb.set_title("Accuracy vs measured truth (the FastFuels gap = within-block)")
    axb.legend(fontsize=8)

    axc = fig.add_subplot(gs[1, 2:])
    cvs = [truth10.std() / truth10.mean()] + [reps[n]["cv"] for n in names]
    labels = ["TRUTH"] + [n.split(" (")[0] for n in names]
    axc.bar(range(len(cvs)), cvs, color=["#444"] + ["#2c7fb8"] * len(names))
    axc.set_xticks(range(len(cvs))); axc.set_xticklabels(labels, fontsize=8, rotation=15)
    axc.set_ylabel("heterogeneity (CV)")
    axc.set_title("Spatial heterogeneity recovered (FastFuels ≈ 0)")
    fig.suptitle(f"{SITE.upper()} end-to-end: spaceborne→30 m→{RES10:.0f} m measured-structure vs FastFuels uniform "
                 f"| Stage-1 R²={rep30['r2']}, end-to-end R²={reps['End-to-end (ours)']['r2']}",
                 fontsize=12, y=0.99)
    fp = os.path.join(ROOT, "figures", f"pipeline_{SITE}.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=115, bbox_inches="tight")
    print(f"\nsaved {fp}")

    np.savez_compressed(os.path.join(ROOT, "data", "processed", f"pipeline_{SITE}.npz"),
                        truth10=truth10, truth30=truth30, pred30=pred30, uniform=uniform,
                        stage1_up=stage1_up, clean=clean["downscaled"], e2e=e2e,
                        canopy30=canopy30, factor=FACTOR, x0=x0, y0=y0, res10=RES10, epsg=TGT_EPSG)
    print(f"Wrote data/processed/pipeline_{SITE}.npz")
    print("\nError budget: Stage-1 carries the 30 m level (spaceborne); Stage-2 adds the within-block "
          "detail FastFuels cannot (its within-block R² is 0). End-to-end propagates Stage-1 error honestly.")


if __name__ == "__main__":
    main()
