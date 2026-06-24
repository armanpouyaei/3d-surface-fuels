"""Real satellite/aerial basemap for the dashboard RGB panel.

Fetches Esri World Imagery (free, no key) for a voxel grid's geographic
footprint, so the "RGB" panel shows the *actual* ground beneath our fuel map
instead of a synthetic render. Used in the dashboard's real-data mode.
"""

from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import numpy as np

ESRI_EXPORT = (
    "https://services.arcgisonline.com/arcgis/rest/services/"
    "World_Imagery/MapServer/export"
)


def _to_webmercator(west, south, east, north, src_epsg):
    """Transform a bbox (in src CRS) to Web Mercator (EPSG:3857) corners."""
    from pyproj import Transformer

    tr = Transformer.from_crs(src_epsg, 3857, always_xy=True)
    xs, ys = tr.transform([west, east, west, east], [south, south, north, north])
    return min(xs), min(ys), max(xs), max(ys)


def fetch_basemap_rgb(
    bounds: Tuple[float, float, float, float],
    src_epsg: int = 32616,
    px: int = 600,
    cache_path: Optional[str] = None,
    timeout: int = 30,
) -> Optional[np.ndarray]:
    """Return an (H, W, 3) uint8 aerial image for ``bounds`` = (W, S, E, N).

    Coordinates are in ``src_epsg`` (default UTM 16N metres). The image is
    fetched in Web Mercator so it is not distorted, and the pixel grid matches
    the bbox aspect. Returns ``None`` (and never raises) if the fetch fails, so
    the dashboard can fall back to the synthetic render offline.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            from PIL import Image

            return np.asarray(Image.open(cache_path).convert("RGB"))
        except Exception:
            pass

    try:
        import requests
        from PIL import Image

        w, s, e, n = _to_webmercator(*bounds, src_epsg)
        aspect = (n - s) / (e - w) if (e - w) else 1.0
        size = f"{px},{max(1, int(round(px * aspect)))}"
        params = {
            "bbox": f"{w},{s},{e},{n}",
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": size,
            "format": "png",
            "f": "image",
        }
        r = requests.get(ESRI_EXPORT, params=params, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        rgb = np.asarray(img)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            img.save(cache_path)
        return rgb
    except Exception:
        return None


def basemap_for_grid(grid, px: int = 600, cache_path: Optional[str] = None):
    """Fetch the aerial basemap covering a ``FuelVoxelGrid``'s footprint."""
    g = grid.georef
    west, south = g.x0, g.y0
    east = g.x0 + grid.nx * grid.dx
    north = g.y0 + grid.ny * grid.dy
    epsg = int(str(g.crs).split(":")[-1]) if ":" in str(g.crs) else 32616
    return fetch_basemap_rgb((west, south, east, north), epsg, px, cache_path)
