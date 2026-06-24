"""3D Surface Fuels — meter-scale voxelized surface/understory fuel mapping.

A small, explainable toolkit for representing, generating, and validating
1 m x 1 m x 1 m voxelized surface-fuel grids for next-generation fire models
(QUIC-Fire, FIRETEC, FDS) and FastFuels-compatible export.
"""

from .voxel import FuelVoxelGrid, GeoRef
from . import metrics, synthetic

__all__ = ["FuelVoxelGrid", "GeoRef", "metrics", "synthetic"]
__version__ = "0.1.0"
