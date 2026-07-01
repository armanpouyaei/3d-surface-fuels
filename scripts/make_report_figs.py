"""Methodology diagrams for the submission Word document:
  fig_method.png       — the end-to-end method flow (measure → predict → field-calibrate → export)
  fig_vs_fastfuels.png — our measured heterogeneous load vs FastFuels' uniform layer (real OSBS data)
  fig_calibration.png  — occupancy → RxCADRE total surface load field calibration (9 Eglin blocks)

Run:  python scripts/make_report_figs.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, GREEN, ORANGE, GREY = "#2c7fb8", "#31a354", "#e6752b", "#555555"


def box(ax, x, y, w, h, text, color, fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.6, edgecolor=color, facecolor=fc or (color + "22")))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color="#111")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color=color))


def fig_method():
    fig, ax = plt.subplots(figsize=(11, 5.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(5.5, 5.15, "Method: measure where LiDAR exists · predict from free spaceborne everywhere · "
            "field-calibrate to load", ha="center", fontsize=11, fontweight="bold")
    # top row: training (LiDAR path)
    box(ax, 0.2, 3.7, 2.1, 1.0, "Airborne LiDAR\n(USGS 3DEP, open)", GREEN)
    box(ax, 2.7, 3.7, 2.3, 1.0, "Vertical OCCUPANCY\n0.15–4 m return-bin\nfill (density-robust)", GREEN)
    arrow(ax, 2.3, 4.2, 2.7, 4.2)
    box(ax, 5.4, 3.7, 2.4, 1.0, "RxCADRE clip plots\n(destructive truth)", ORANGE)
    box(ax, 8.2, 3.7, 2.6, 1.0, "occupancy → LOAD\nK=1.33 kg/m², R²=0.93", ORANGE)
    arrow(ax, 5.0, 4.2, 5.4, 4.2); arrow(ax, 7.8, 4.2, 8.2, 4.2)
    # middle: spaceborne inputs
    box(ax, 0.2, 1.7, 4.8, 1.2,
        "GLOBAL FREE spaceborne inputs\nAlphaEarth 64-band · Sentinel-1 C-band ·\nALOS PALSAR L-band · global canopy height",
        BLUE)
    box(ax, 5.4, 1.9, 2.4, 0.9, "Per-pixel quantile\nGradient Boosting", BLUE)
    box(ax, 8.2, 1.9, 2.6, 0.9, "predicted occupancy\n+ conformal band + OOD", BLUE)
    arrow(ax, 5.0, 2.35, 5.4, 2.35); arrow(ax, 7.8, 2.35, 8.2, 2.35)
    arrow(ax, 3.5, 3.7, 3.6, 2.9, GREEN)  # occupancy target trains the model
    ax.text(3.75, 3.25, "trains", fontsize=8, color=GREEN, rotation=90, va="center")
    # bottom: output
    box(ax, 2.7, 0.2, 5.6, 1.0,
        "1 m³ FastFuels-compatible NetCDF (Option C)\nbulk density · load · % cover · SAVR · live/dead · moisture · + boundary GeoJSON",
        "#7b3294")
    arrow(ax, 9.5, 1.9, 6.5, 1.2, "#7b3294")   # predicted -> output
    arrow(ax, 9.5, 3.7, 6.2, 1.2, ORANGE)      # calibration -> output
    fig.tight_layout(); p = os.path.join(FIG, "fig_method.png"); fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_vs_fastfuels():
    import xarray as xr
    try:
        ours = xr.open_dataset(os.path.join(ROOT, "data/processed/osbs_measured_1m.nc"))["fuel_load"].values
        unif = xr.open_dataset(os.path.join(ROOT, "data/processed/osbs_uniform_1m.nc"))["fuel_load"].values
    except Exception:
        return None
    vmax = float(np.percentile(ours, 99))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    for a, arr, t in [(ax[0], unif, "FastFuels-style: uniform per class\n(CV = 0 — no sub-class detail)"),
                      (ax[1], ours, "Ours: measured 1 m surface load\n(CV = 1.13 — real heterogeneity)")]:
        im = a.imshow(np.flipud(arr), cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
        a.set_title(t, fontsize=10.5); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="surface fuel load (kg/m²)")
    fig.suptitle("OSBS 510×510 m — same fuel load, two products", fontsize=12, y=0.98)
    p = os.path.join(FIG, "fig_vs_fastfuels.png"); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def fig_calibration():
    # 9 Eglin RxCADRE blocks: (model occupancy, field total load kg/m²) from calibrate_load_field.py
    occ = np.array([0.23, 0.78, 0.26, 0.25, 0.25, 0.25, 0.24, 0.25, 0.22])
    load = np.array([0.21, 1.12, 0.27, 0.31, 0.25, 0.28, 0.41, 0.36, 0.24])
    K = 1.33
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(occ, load, s=70, color=ORANGE, edgecolor="k", lw=0.5, zorder=3, label="Eglin burn blocks (n=9)")
    xx = np.linspace(0, 0.85, 50); ax.plot(xx, K * xx, "--", color=GREY, lw=1.6,
            label=f"field fit: load = {K}·occ  (R²=0.93)")
    ax.set_xlabel("model vertical occupancy (0–1)"); ax.set_ylabel("RxCADRE total surface load (kg/m²)")
    ax.set_title("Field calibration to destructive truth\n(occupancy → total surface load, Eglin)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 0.85); ax.set_ylim(0, 1.25)
    p = os.path.join(FIG, "fig_calibration.png"); fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_method(), fig_vs_fastfuels(), fig_calibration()):
        print("wrote", os.path.relpath(f, ROOT) if f else "(skipped — deliverable NetCDFs missing)")
