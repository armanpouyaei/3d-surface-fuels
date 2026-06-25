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

# (label, lat, lon, in_training_region?)  — small AOIs keep AEF fetches quick
POINTS = [
    ("Germany (Harz)",        51.75,  10.60, "EU-new"),
    ("UK (Midlands)",         52.50,  -1.50, "EU-new"),
    ("Spain (Castile)",       40.50,  -4.00, "EU-new"),
    ("Sweden (boreal)",       60.20,  15.60, "EU-new"),
    ("Switzerland*",          46.80,   8.20, "train"),
    ("France*",               45.20,   6.10, "train"),
    ("Netherlands*",          52.10,   5.80, "train"),
    ("OSBS Florida* (US)",    29.68, -82.00, "train"),
    ("SRER Arizona* (US)",    31.82,-110.87, "train"),
    ("Amazon (Brazil)",       -3.10, -60.00, "foreign"),
    ("Sahara (Niger)",        18.00,  10.00, "foreign"),
    ("Outback (Australia)",  -25.00, 133.00, "foreign"),
    ("Congo basin",           -1.00,  23.00, "foreign"),
]

print(f"{'region':<24}{'kind':<9}{'frac_OOD':>9}   verdict")
print("-" * 60)
for label, lat, lon, kind in POINTS:
    try:
        r = portable.predict_aoi(lat, lon, size=300.0, year=2022, res=10,
                                 use_cache=True, version="v2")
        f = r["frac_ood"]
        verdict = "IN-dist" if f < 0.5 else ("OOD" if f > 0.9 else "borderline")
        print(f"{label:<24}{kind:<9}{f:>9.2f}   {verdict}", flush=True)
    except Exception as e:
        print(f"{label:<24}{kind:<9}{'n/a':>9}   fetch failed: {type(e).__name__} {str(e)[:60]}", flush=True)
