"""WHY does cross-ecosystem LOSO R² sit at ~0? Diagnose on the cached 10-site grids,
no downloads. Tests three hypotheses:
  H1  the per-site target normalization (->mean 0.6) erased the between-biome signal
      => between-site variance of the target is ~0 by construction.
  H2  AlphaEarth DOES encode biome (so a between-biome model is feasible if we keep that
      variance) => pixelwise site-classification accuracy from AEF is high.
  H3  the residual *within-biome* structure is the genuinely hard part (sensing limit).
Run:  python scripts/diagnose_transfer.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        site = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = np.load(f); out[site] = {k: d[k] for k in d.files}
    return out


def main():
    grids = load_grids()
    sites = list(grids)
    print(f"sites ({len(sites)}): {sites}\n")

    # ---- H1: per-site target stats + between/within variance decomposition ----
    print("H1  target per-site mean/std (normalization erases between-biome scale?)")
    ys, labels = [], []
    means = {}
    for i, s in enumerate(sites):
        g = grids[s]; v = g["valid"].ravel(); y = g["target"].ravel()[v]
        means[s] = y.mean()
        ys.append(y); labels.append(np.full(len(y), i))
        print(f"   {s:5}: mean {y.mean():.3f}  std {y.std():.3f}  cv {y.std()/max(y.mean(),1e-6):.2f}")
    Y = np.concatenate(ys); L = np.concatenate(labels)
    grand = Y.mean()
    ss_between = sum(len(grids[s]["target"].ravel()[grids[s]["valid"].ravel()]) * (means[s] - grand) ** 2
                     for s in sites)
    ss_total = ((Y - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    print(f"   --> between-site variance fraction (eta²) = {eta2:.3f}  "
          f"({'erased' if eta2 < 0.05 else 'present'}); grand mean {grand:.3f}\n")

    # ---- H2: does AEF encode biome?  pixelwise site classification (random split) ----
    print("H2  AlphaEarth -> biome separability (random 70/30 split, HGB classifier)")
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ls = [], []
    for i, s in enumerate(sites):
        g = grids[s]; v = g["valid"].ravel()
        A = np.stack([np.nan_to_num(g["aef"][k]).ravel() for k in range(64)], 1)[v]
        # subsample for speed
        idx = np.linspace(0, len(A) - 1, min(4000, len(A))).astype(int)
        Xs.append(A[idx]); ls.append(np.full(len(idx), i))
    Xc = np.vstack(Xs); lc = np.concatenate(ls)
    rng = np.random.RandomState(0); perm = rng.permutation(len(Xc))
    cut = int(0.7 * len(Xc)); tr, te = perm[:cut], perm[cut:]
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=200)
    clf.fit(Xc[tr], lc[tr])
    acc = (clf.predict(Xc[te]) == lc[te]).mean()
    print(f"   --> AEF predicts which of {len(sites)} biomes a pixel is in: acc {acc:.3f} "
          f"(chance {1/len(sites):.3f})  => AEF {'STRONGLY' if acc>0.7 else 'weakly'} encodes biome\n")

    # ---- H3: how much within-site structure variance is there to explain at all? ----
    print("H3  within-site spatial signal (target CV per site, already above) is the hard residual.")
    print(f"   mean within-site CV = {np.mean([grids[s]['target'].ravel()[grids[s]['valid'].ravel()].std()/max(means[s],1e-6) for s in sites]):.2f}")
    print("\nIMPLICATION: if eta²~0 and AEF-biome-acc is high, the cross-ecosystem signal was")
    print("thrown away by normalization, NOT unlearnable. Fix = target that keeps between-biome scale.")


if __name__ == "__main__":
    main()
