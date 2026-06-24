"""Multi-ecosystem field-calibrated Stage-1 analysis.

Pools per-site AlphaEarth feature tables (data/interim/field_<site>.csv) and tests:
  * within-site skill  — leave-one-PLOT-out per site (can AEF predict herb load here?)
  * GENERALITY         — leave-one-SITE-out (train on other ecosystems, predict an
                         unseen one) — the competition's key axis.
Figures + metrics saved/printed. Usage: python scripts/analyze_field_multisite.py
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

from surface_fuels import metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
AEF = [f"aef{i:02d}" for i in range(64)]


def model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def lopo(d, feats):
    X, y, g = d[feats].values, d.load_kg_m2.values, d.plotID.values
    pred = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return np.clip(pred, 0, None)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "interim", "field_*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    feats = [c for c in AEF if c in d.columns]
    sites = sorted(d.site.unique())
    print(f"{len(d)} clips across {len(sites)} sites: {sites}; features: {len(feats)} AEF bands\n")

    # within-site LOPO
    within = {}
    for s in sites:
        ds = d[d.site == s]
        p = lopo(ds, feats)
        within[s] = (M.r2(p, ds.load_kg_m2.values), spearmanr(p, ds.load_kg_m2.values).correlation,
                     ds.load_kg_m2.mean(), len(ds), p)

    # leave-one-SITE-out (generality)
    transfer = {}
    for s in sites:
        tr, te = d[d.site != s], d[d.site == s]
        m = model().fit(tr[feats].values, tr.load_kg_m2.values)
        p = np.clip(m.predict(te[feats].values), 0, None)
        transfer[s] = (M.r2(p, te.load_kg_m2.values), spearmanr(p, te.load_kg_m2.values).correlation)

    print(f"{'site':<8}{'n':>4}{'load µ':>9}{'within R2':>11}{'within ρ':>10}{'transfer ρ':>12}")
    for s in sites:
        w = within[s]; t = transfer[s]
        print(f"{s:<8}{w[3]:>4}{w[2]:>9.3f}{w[0]:>11.2f}{w[1]:>10.2f}{t[1]:>12.2f}")
    # pooled transfer rank correlation
    allp = np.concatenate([np.clip(model().fit(d[d.site != s][feats], d[d.site != s].load_kg_m2)
                                   .predict(d[d.site == s][feats]), 0, None) for s in sites])
    ally = np.concatenate([d[d.site == s].load_kg_m2.values for s in sites])
    print(f"\nPooled leave-one-site-out transfer: Spearman {spearmanr(allp, ally).correlation:.2f} "
          f"(generality across unseen ecosystems)")

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].boxplot([d[d.site == s].load_kg_m2 for s in sites], labels=sites)
    ax[0].set_ylabel("field herb load (kg/m²)"); ax[0].set_title("Ecosystem diversity (herb load)")
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    for s, c in zip(sites, colors):
        w = within[s]; ds = d[d.site == s]
        ax[1].scatter(ds.load_kg_m2, w[4], s=28, alpha=.6, color=c, label=f"{s} ρ={w[1]:.2f}")
    lim = [0, d.load_kg_m2.max()]; ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel("field herb load"); ax[1].set_ylabel("predicted (within-site LOPO)")
    ax[1].set_title("Within-site skill"); ax[1].legend(fontsize=7)
    x = np.arange(len(sites)); wbar = [within[s][1] for s in sites]; tbar = [transfer[s][1] for s in sites]
    ax[2].bar(x - 0.2, wbar, 0.4, label="within-site ρ"); ax[2].bar(x + 0.2, tbar, 0.4, label="transfer ρ (unseen)")
    ax[2].set_xticks(x); ax[2].set_xticklabels(sites); ax[2].axhline(0, color="k", lw=.6)
    ax[2].set_ylabel("Spearman ρ"); ax[2].set_title("Within-site vs leave-one-site-out (generality)"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    fp = os.path.join(ROOT, "figures", "field_multisite.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=110)
    print(f"saved {fp}")


if __name__ == "__main__":
    main()
