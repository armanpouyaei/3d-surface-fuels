"""Voxelized surface-fuel grid: the core data structure.

A `FuelVoxelGrid` is a 3D array of fuel **bulk density** (kg / m^3) on a regular
grid of 1 m (default) voxels, plus optional per-voxel fuel-property arrays
(live/dead fraction, surface-area-to-volume ratio, fuel moisture, species code).

This mirrors the quantities the challenge asks for and is directly exportable to
the FastFuels v2 "Option C" 3D voxelized NetCDF format (cell bulk density in
kg/m^3 at 1 m x 1 m x 1 m), as well as to QUIC-Fire / FIRETEC inputs.

Axis convention: arrays are indexed ``[z, y, x]`` (z = height above ground,
increasing upward; y = northing; x = easting), which is the natural order for
NetCDF and for vertical-profile operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# The challenge says cells with no fuel should default their non-mass values to
# this sentinel so judges can distinguish "measured zero" from "no data".
NO_FUEL_SENTINEL = 1.23456


@dataclass
class GeoRef:
    """Minimal georeferencing for a voxel grid (cell-centered origin).

    The grid's (x=0, y=0) voxel *center* sits at ``(x0, y0)`` in the given CRS.
    """

    crs: str = "EPSG:32616"  # UTM 16N (Florida / SE US default); change per site
    x0: float = 0.0          # easting (m) of the x=0 column center
    y0: float = 0.0          # northing (m) of the y=0 row center
    resolution: float = 1.0  # horizontal voxel size (m)

    def as_attrs(self) -> Dict[str, object]:
        return {
            "crs": self.crs,
            "origin_x": self.x0,
            "origin_y": self.y0,
            "horizontal_resolution_m": self.resolution,
        }


@dataclass
class FuelVoxelGrid:
    """A 1 m^3 voxel grid of surface-fuel bulk density and properties.

    Parameters
    ----------
    bulk_density : np.ndarray
        Shape ``(nz, ny, nx)``, units kg/m^3. The primary required quantity.
    dz, dy, dx : float
        Voxel dimensions in metres (default 1 m each).
    georef : GeoRef
        Horizontal georeferencing.
    extra : dict[str, np.ndarray]
        Optional per-voxel arrays of the same shape (e.g. ``live_fraction``
        [0-1], ``sav`` [m^2/m^3], ``fuel_moisture`` [%], ``species`` [int code]).
    attrs : dict
        Free-form metadata (site name, ecosystem, source, provenance).
    """

    bulk_density: np.ndarray
    dz: float = 1.0
    dy: float = 1.0
    dx: float = 1.0
    georef: GeoRef = field(default_factory=GeoRef)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)
    attrs: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bulk_density = np.asarray(self.bulk_density, dtype=np.float32)
        if self.bulk_density.ndim != 3:
            raise ValueError("bulk_density must be 3D (nz, ny, nx)")
        for name, arr in self.extra.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != self.bulk_density.shape:
                raise ValueError(f"extra['{name}'] shape {arr.shape} != {self.shape}")
            self.extra[name] = arr

    # -- shape helpers --------------------------------------------------
    @property
    def shape(self):
        return self.bulk_density.shape

    @property
    def nz(self):
        return self.bulk_density.shape[0]

    @property
    def ny(self):
        return self.bulk_density.shape[1]

    @property
    def nx(self):
        return self.bulk_density.shape[2]

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def heights(self) -> np.ndarray:
        """Height (m) of each z-layer center above ground."""
        return (np.arange(self.nz) + 0.5) * self.dz

    # -- derived fuel products -----------------------------------------
    def fuel_load(self) -> np.ndarray:
        """2D fuel loading (kg/m^2): bulk density integrated over height."""
        return self.bulk_density.sum(axis=0) * self.dz

    def fuelbed_depth(self) -> np.ndarray:
        """2D fuelbed depth (m): top of the highest occupied voxel per column."""
        occ = self.bulk_density > 0
        # index of highest occupied layer (+1 -> top face), 0 where empty
        has_fuel = occ.any(axis=0)
        top_idx = np.where(occ.any(axis=0), occ.shape[0] - 1 -
                           np.argmax(occ[::-1, :, :], axis=0), -1)
        depth = (top_idx + 1) * self.dz
        depth[~has_fuel] = 0.0
        return depth.astype(np.float32)

    def occupancy(self, threshold: float = 0.0) -> np.ndarray:
        """Boolean voxel occupancy: bulk density strictly above ``threshold``."""
        return self.bulk_density > threshold

    def vertical_profile(self) -> np.ndarray:
        """Mean bulk density (kg/m^3) per height layer, averaged over x,y."""
        return self.bulk_density.mean(axis=(1, 2))

    def total_mass(self) -> float:
        """Total fuel mass (kg) in the domain."""
        return float(self.bulk_density.sum() * self.cell_volume)

    def mean_load(self) -> float:
        """Mean fuel loading (kg/m^2) over the domain."""
        return float(self.fuel_load().mean())

    # -- xarray / NetCDF I/O -------------------------------------------
    def to_dataset(self):
        """Return an xarray.Dataset with coordinates in metres + CRS attrs."""
        import xarray as xr

        z = self.heights()
        y = self.georef.y0 + (np.arange(self.ny) + 0.5) * self.dy
        x = self.georef.x0 + (np.arange(self.nx) + 0.5) * self.dx

        data_vars = {
            "bulk_density": (
                ("z", "y", "x"),
                self.bulk_density,
                {"units": "kg m-3", "long_name": "fuel bulk density"},
            )
        }
        for name, arr in self.extra.items():
            data_vars[name] = (("z", "y", "x"), arr)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "z": ("z", z, {"units": "m", "long_name": "height above ground"}),
                "y": ("y", y, {"units": "m", "long_name": "northing"}),
                "x": ("x", x, {"units": "m", "long_name": "easting"}),
            },
            attrs={
                **self.georef.as_attrs(),
                "voxel_dz_m": self.dz,
                "voxel_dy_m": self.dy,
                "voxel_dx_m": self.dx,
                "no_fuel_sentinel": NO_FUEL_SENTINEL,
                "Conventions": "CF-1.8",
                "title": "3D surface fuels voxel grid",
                **self.attrs,
            },
        )
        return ds

    def to_netcdf(self, path: str) -> None:
        """Write a FastFuels-compatible (Option C) NetCDF file."""
        self.to_dataset().to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str) -> "FuelVoxelGrid":
        import xarray as xr

        ds = xr.open_dataset(path)
        bd = ds["bulk_density"].values
        extra = {
            v: ds[v].values
            for v in ds.data_vars
            if v != "bulk_density" and ds[v].dims == ("z", "y", "x")
        }
        georef = GeoRef(
            crs=str(ds.attrs.get("crs", "EPSG:32616")),
            x0=float(ds.attrs.get("origin_x", 0.0)),
            y0=float(ds.attrs.get("origin_y", 0.0)),
            resolution=float(ds.attrs.get("horizontal_resolution_m", 1.0)),
        )
        return cls(
            bulk_density=bd,
            dz=float(ds.attrs.get("voxel_dz_m", 1.0)),
            dy=float(ds.attrs.get("voxel_dy_m", 1.0)),
            dx=float(ds.attrs.get("voxel_dx_m", 1.0)),
            georef=georef,
            extra=extra,
            attrs={k: v for k, v in ds.attrs.items()},
        )

    def summary(self) -> Dict[str, float]:
        """Quick scalar summary for the 'smell test'."""
        load = self.fuel_load()
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "total_mass_kg": round(self.total_mass(), 1),
            "mean_load_kg_m2": round(float(load.mean()), 3),
            "max_load_kg_m2": round(float(load.max()), 3),
            "load_cv": round(float(load.std() / (load.mean() + 1e-9)), 3),
            "occupied_fraction": round(float(self.occupancy().mean()), 4),
            "mean_depth_m": round(float(self.fuelbed_depth().mean()), 3),
        }
