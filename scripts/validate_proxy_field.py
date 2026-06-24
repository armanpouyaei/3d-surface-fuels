"""Biggest-risk check: does the 3DEP near-ground return-density PROXY (our
wall-to-wall Stage-1 target) track REAL NEON field fuel load?

Builds the proxy at the NEON OSBS herb-clip locations from the covering 3DEP
tiles and correlates proxy vs field herb load. If they correlate, the proxy
target is defensible; if not, the "surface fuel" framing must be re-scoped.

Usage: python scripts/validate_proxy_field.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from surface_fuels import lidar  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
EPSG = 32617
RES = 30.0
WIN_BUF = 150.0


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "interim", "neon_osbs_herb.csv"))
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    d["x"], d["y"] = tr.transform(d.decimalLongitude.values, d.decimalLatitude.values)
    x0, y0 = d.x.min() - WIN_BUF, d.y.min() - WIN_BUF
    X1, Y1 = d.x.max() + WIN_BUF, d.y.max() + WIN_BUF
    nx = int((X1 - x0) // RES) + 1
    ny = int((Y1 - y0) // RES) + 1

    tiles = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "osbs_clip_tile_*.laz")))
    print(f"combining {len(tiles)} 3DEP tiles over the clip window ({nx}x{ny} @ 30 m)...")
    xs, ys, zs, cs = [], [], [], []
    for t in tiles:
        pc = lidar.load_points(t, src_epsg=6438, target_epsg=EPSG, z_unit="ft_us")
        m = (pc.x >= x0) & (pc.x < X1) & (pc.y >= y0) & (pc.y < Y1)
        if m.any():
            xs.append(pc.x[m]); ys.append(pc.y[m]); zs.append(pc.z[m]); cs.append(pc.cls[m])
        del pc
    from surface_fuels.lidar import PointCloud
    pc = PointCloud(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                    np.concatenate(cs), EPSG)
    print(f"  {len(pc.x):,} window points")

    # ground DTM + near-ground (0.15-4 m) return-density proxy per 30 m cell
    dtm, dres, ng = lidar._ground_dtm(pc, x0, y0, max(nx, ny) * RES, res=5.0)
    gi = np.clip(((pc.x - x0) / dres).astype(int), 0, ng - 1)
    gj = np.clip(((pc.y - y0) / dres).astype(int), 0, ng - 1)
    h = pc.z - dtm[gj, gi]
    ix = np.clip(((pc.x - x0) / RES).astype(int), 0, nx - 1)
    iy = np.clip(((pc.y - y0) / RES).astype(int), 0, ny - 1)
    surf = (h >= 0.15) & (h < 4.0)
    proxy = np.zeros((ny, nx), np.float32)
    np.add.at(proxy, (iy[surf], ix[surf]), 1.0)
    tot = np.zeros((ny, nx), np.float32); np.add.at(tot, (iy, ix), 1.0)
    proxy = np.divide(proxy, tot, out=np.zeros_like(proxy), where=tot > 0)  # near-ground return fraction

    # sample proxy at each clip
    cx = np.clip(((d.x - x0) / RES).astype(int), 0, nx - 1).values
    cy = np.clip(((d.y - y0) / RES).astype(int), 0, ny - 1).values
    d["proxy"] = proxy[cy, cx]
    d = d[tot[cy, cx] > 0]  # clips with 3DEP coverage

    from scipy.stats import pearsonr, spearmanr
    from surface_fuels import metrics as M
    pr = pearsonr(d.proxy, d.load_kg_m2); sp = spearmanr(d.proxy, d.load_kg_m2)
    a, b = np.polyfit(d.proxy, d.load_kg_m2, 1)
    print(f"\n3DEP near-ground proxy vs NEON field herb load (n={len(d)} clips with coverage):")
    print(f"  Pearson r {pr[0]:+.2f} (p={pr[1]:.3f}) | Spearman {sp.correlation:+.2f} | "
          f"R² {M.r2(a*d.proxy.values+b, d.load_kg_m2.values):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(d.proxy, d.load_kg_m2, s=45, alpha=.6, color="#7b3294", edgecolor="k", lw=.3)
    xx = np.linspace(d.proxy.min(), d.proxy.max(), 50); ax[0].plot(xx, a * xx + b, "r--", lw=1)
    ax[0].set_xlabel("3DEP near-ground return fraction (proxy)"); ax[0].set_ylabel("NEON field herb load (kg/m²)")
    ax[0].set_title(f"Proxy vs field: Pearson {pr[0]:.2f}, Spearman {sp.correlation:.2f}")
    im = ax[1].imshow(proxy, origin="lower", cmap="YlGn", extent=[x0, X1, y0, Y1], aspect="equal")
    ax[1].scatter(d.x, d.y, c=d.load_kg_m2, cmap="autumn_r", s=35, edgecolor="k", lw=.4)
    ax[1].set_title("Proxy map + clips (color=field load)"); plt.colorbar(im, ax=ax[1], label="proxy")
    plt.tight_layout()
    fp = os.path.join(ROOT, "figures", "proxy_vs_field.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=110)
    print(f"saved {fp}")
    print("Note: proxy is near-ground (0.15-4 m) ALL vegetation; field is herbaceous only — "
          "so a moderate (not perfect) correlation is expected and honest.")


if __name__ == "__main__":
    main()
