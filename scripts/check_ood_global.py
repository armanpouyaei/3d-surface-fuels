"""Did broadening training coverage (now 10 sites incl. Switzerland/Netherlands/France)
close the European OOD gap? Run the v2 model over small AOIs at a spread of global points
and report the out-of-distribution fraction. Europe should now be IN-distribution; clearly
foreign biomes (Amazon, Sahara, outback) should still flag high so the OOD guard stays honest.

Usage:  python scripts/check_ood_global.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surface_fuels import portable  # noqa: E402

# (label, lat, lon, kind) — small AOIs keep AEF fetches quick. After adding tropical (PR),
# boreal (Latvia) and arid (Mojave) training, the once-foreign biomes should move in-distribution.
POINTS = [
    ("Germany (temperate)",   51.75,  10.60, "EU"),
    ("Sweden (boreal)",       60.20,  15.60, "boreal-new"),
    ("Finland (boreal)",      62.00,  26.00, "boreal-new"),
    ("Canada (boreal)",       54.00, -98.00, "boreal-new"),
    ("Amazon (tropical)",     -3.10, -60.00, "tropical-new"),
    ("Congo (tropical)",      -1.00,  23.00, "tropical-new"),
    ("SE Asia (tropical)",     0.50, 113.00, "tropical-new"),
    ("Sahara (Niger)",        18.00,  10.00, "arid-new"),
    ("Outback (Australia)",  -25.00, 133.00, "arid-new"),
    ("Arabia (desert)",       23.00,  45.00, "arid-new"),
    ("OSBS Florida* (US)",    29.68, -82.00, "train"),
    ("Puerto Rico* (trop)",   18.14, -65.50, "train"),
    ("Mojave* (arid)",        35.90,-115.12, "train"),
]

print(f"{'region':<24}{'kind':<9}{'frac_OOD':>9}   verdict")
print("-" * 60)
for label, lat, lon, kind in POINTS:
    try:
        r = portable.predict_aoi(lat, lon, size=300.0, year=2022, res=10,
                                 use_cache=True, version="v3")
        f = r["frac_ood"]
        verdict = "IN-dist" if f < 0.5 else ("OOD" if f > 0.9 else "borderline")
        print(f"{label:<24}{kind:<9}{f:>9.2f}   {verdict}", flush=True)
    except Exception as e:
        print(f"{label:<24}{kind:<9}{'n/a':>9}   fetch failed: {type(e).__name__} {str(e)[:60]}", flush=True)
