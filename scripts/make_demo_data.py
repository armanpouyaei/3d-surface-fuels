"""Generate the synthetic demo scene, save NetCDF voxel grids, and print a
validation report. Run: ``python scripts/make_demo_data.py``.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surface_fuels import metrics  # noqa: E402
from surface_fuels.synthetic import make_demo_scenario  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    os.makedirs(OUT, exist_ok=True)
    scene = make_demo_scenario(nx=100, ny=100, nz=4, seed=42)

    for name, grid in (("truth", scene.truth),
                       ("fastfuels", scene.fastfuels),
                       ("ours", scene.ours)):
        path = os.path.join(OUT, f"{name}.nc")
        grid.to_netcdf(path)
        print(f"wrote {path}")
        print("  summary:", json.dumps(grid.summary()))

    print("\n=== FastFuels baseline vs truth ===")
    print(json.dumps(metrics.compare(scene.fastfuels, scene.truth), indent=2))
    print("\n=== OUR method vs truth ===")
    print(json.dumps(metrics.compare(scene.ours, scene.truth), indent=2))


if __name__ == "__main__":
    main()
