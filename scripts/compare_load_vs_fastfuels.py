"""Is our predicted surface load systematically lower than FastFuels? For several US AOIs
(LANDFIRE/FastFuels is US-only) compare mean absolute load:
  ours = v3 occupancy x OCC_TO_LOAD_KG_M2   vs   FastFuels = LANDFIRE FBFM40 -> SB40 load.
Reports both means + ratio, to see if/why ours is low.

Run:  python scripts/compare_load_vs_fastfuels.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from surface_fuels import portable  # noqa: E402

K = portable.OCC_TO_LOAD_KG_M2
REGIONS = [  # US sites/biomes with LANDFIRE coverage
    ("OSBS savanna",   29.68,  -82.00),
    ("SOAP conifer",   37.03, -119.26),
    ("WREF conifer",   45.82, -121.95),
    ("HARV decid.",    42.54,  -72.17),
    ("CPER grass",     40.82, -104.75),
    ("SRER desert",    31.82, -110.87),
]


def main():
    print(f"OCC_TO_LOAD_KG_M2 = {K:.3f}\n")
    print(f"{'region':<16}{'occ μ':>7}{'ours kg/m²':>12}{'FastFuels kg/m²':>17}{'ours/FF':>9}")
    print("-" * 61)
    ratios = []
    for name, lat, lon in REGIONS:
        try:
            r = portable.predict_aoi(lat, lon, size=300.0, year=2022, res=10, use_cache=True, version="v3")
            occ = r["pred10"]
            ours = float(occ.mean()) * K
            ff, lfy = portable.fastfuels_surface_aoi(r, lat, lon, 2022)
            if ff is None:
                print(f"{name:<16}{occ.mean():>7.2f}{ours:>12.2f}{'n/a':>17}{'—':>9}"); continue
            ffm = float(ff[ff > 0].mean()) if (ff > 0).any() else float("nan")
            ratio = ours / ffm if ffm else float("nan")
            ratios.append(ratio)
            print(f"{name:<16}{occ.mean():>7.2f}{ours:>12.2f}{ffm:>17.2f}{ratio:>9.2f}  (LF{lfy})", flush=True)
        except Exception as e:
            print(f"{name:<16} failed: {type(e).__name__} {str(e)[:50]}", flush=True)
    if ratios:
        print(f"\nmean ours/FastFuels ratio = {np.mean(ratios):.2f}  "
              f"({'ours LOWER' if np.mean(ratios) < 0.9 else 'comparable' if np.mean(ratios) < 1.1 else 'ours HIGHER'})")
    print("Note: SB40 surface loads reach ~2+ kg/m² for heavy models; our occ×K is capped at ~1 kg/m² (occ≤1).")


if __name__ == "__main__":
    main()
