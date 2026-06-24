"""30 m -> 1 m surface-fuel downscaler POC (Week 3).

Demonstrates that a regression downscaler recovers the sub-30 m heterogeneity a
naive coarse upsample cannot — the core of the *global* product idea: a coarse
spaceborne baseline (available everywhere) sharpened to 1 m by globally-available
10 m covariates, trained on US airborne-LiDAR targets.

Validated by spatially-blocked (quadrant) cross-validation; mass-conserving.

Usage:  python scripts/run_downscale_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surface_fuels import downscale as ds  # noqa: E402
from surface_fuels.synthetic import make_downscale_scene  # noqa: E402


def main():
    sc = make_downscale_scene(n=240, seed=7, factor=30)
    coarse = ds.block_coarsen(sc.truth, sc.factor)
    print(f"Scene 240x240 @ 1 m; coarse baseline {coarse.shape} (~{sc.factor} m); truth CV {sc.truth.std()/sc.truth.mean():.3f}")
    print("Covariates (globally available 10 m): Sentinel-1, multi-sensor/embedding, canopy\n")

    out = ds.downscale_cv(sc.truth, coarse,
                          {"s1": sc.s1, "embed": sc.embed, "canopy": sc.canopy}, sc.factor)
    rep = ds.evaluate(out, sc.truth, sc.factor)

    hdr = f"{'method':<22}{'R2':>8}{'within-block R2':>18}{'CV':>8}{'agg-consist':>13}"
    print(hdr); print("-" * len(hdr))
    for k, lab in [("coarse_baseline", "coarse upsample (30 m)"), ("downscaler", "downscaler (1 m)")]:
        r = rep[k]
        print(f"{lab:<22}{r['r2']:>8}{r['within_block_r2']:>18}{r['cv']:>8}{r['agg_consistency_r2']:>13}")

    d, b = rep["downscaler"], rep["coarse_baseline"]
    print(f"\nDownscaler lifts overall R2 by {d['r2']-b['r2']:+.2f} and recovers within-block "
          f"(sub-30 m) skill {b['within_block_r2']} -> {d['within_block_r2']} that the coarse "
          f"upsample structurally cannot, while conserving mass (agg R2 {d['agg_consistency_r2']}).")
    if out["feature_importance"] is not None:
        imp = sorted(zip(out["feature_names"], out["feature_importance"]), key=lambda t: -t[1])
        print("Feature importance:", {n: round(float(v), 2) for n, v in imp[:5]})
    print("\nThe finest <10 m band is information-limited from 10 m covariates — which is why "
          "the ALS-backed 1 m product stays the gold standard where airborne LiDAR exists. "
          "The 10 m embedding covariate dominates; AlphaEarth/Clay embeddings + a UNet are the "
          "upgrade path (see research/DOWNSCALING.md).")


if __name__ == "__main__":
    main()
