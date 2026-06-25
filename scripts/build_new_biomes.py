"""Build base grids (AEF+S1+normalized target) for the 3 new biome sites (tropical PR,
boreal Latvia, arid Mojave). After this, run rederive_targets.py, rederive_thinned.py,
add_physical_features.py, add_canopy.py to fill the rest of the features (they skip done sites).

Run:  python scripts/build_new_biomes.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_portable import SITES, ROOT, site_samples  # noqa: E402

NEW = {"pr", "lva", "mojave"}

for s in SITES:
    if s["name"] not in NEW:
        continue
    gp = os.path.join(ROOT, "data", "interim", f"portable_grid_{s['name']}.npz")
    if os.path.exists(gp):
        print(f"[{s['name']}] grid cached — skip", flush=True); continue
    if not os.path.exists(os.path.join(ROOT, s["laz"])):
        print(f"[{s['name']}] LAZ missing ({s['laz']}) — skip", flush=True); continue
    try:
        site_samples(s)
    except Exception as e:
        print(f"[{s['name']}] build FAILED: {type(e).__name__} {str(e)[:140]}", flush=True)
print("base build done.")
