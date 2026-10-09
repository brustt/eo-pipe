"""Sentinel-2 MGRS Tile lookup.

A Tile is a 109.8 km square in its own UTM zone.  Real Sentinel-2 tile origins
carry small (multiples of 20 m) offsets from the nominal MGRS 100 km corner, so
the exact origin is taken from the grid footprint reprojected to the tile's UTM
CRS and rounded to 20 m — never derived from the tile name alone.  The name only
gives the UTM zone and hemisphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd
from pyproj import Transformer
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

#: Default Sentinel-2 Tile grid (ocean-masked, EPSG:4326 footprints, ``Name`` column).
S2_TILE_GRID_PATH = Path("/home/brustt/Documents/Projects/data/s2_mgrs_worldwide_grid_land.parquet")

#: Side of a Sentinel-2 Tile in metres.
TILE_SIZE_M = 109_800.0

_ORIGIN_SNAP_M = 20.0
_SOUTHERN_BANDS = "CDEFGHJKLM"


@dataclass(frozen=True)
class Tile:
    """A Sentinel-2 Tile: name, UTM CRS and upper-left corner of its lattice."""

    name: str
    crs: CRS
    ul_x: float
    ul_y: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` in the tile CRS."""
        return (
            self.ul_x,
            self.ul_y - TILE_SIZE_M,
            self.ul_x + TILE_SIZE_M,
            self.ul_y,
        )


def tile_epsg(name: str) -> int:
    """EPSG code of the UTM zone of tile *name* (zone and hemisphere from the name)."""
    zone = int(name[:2])
    base = 32700 if name[2].upper() in _SOUTHERN_BANDS else 32600
    return base + zone


@lru_cache(maxsize=4)
def _load_grid(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    if "Name" not in gdf.columns:
        raise ValueError(f"Tile grid {path} has no 'Name' column.")
    return gdf.to_crs(4326) if gdf.crs is not None and gdf.crs.to_epsg() != 4326 else gdf


def _tile_from_footprint(name: str, footprint: BaseGeometry) -> Tile:
    epsg = tile_epsg(name)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    minx, _, _, maxy = shp_transform(to_utm, footprint).bounds
    return Tile(
        name=name,
        crs=CRS.from_epsg(epsg),
        ul_x=round(minx / _ORIGIN_SNAP_M) * _ORIGIN_SNAP_M,
        ul_y=round(maxy / _ORIGIN_SNAP_M) * _ORIGIN_SNAP_M,
    )


def find_tiles(footprint: BaseGeometry, grid_path: Union[str, Path] = S2_TILE_GRID_PATH) -> List[Tile]:
    """Tiles of the grid whose footprint intersects *footprint* (EPSG:4326)."""
    grid = _load_grid(str(grid_path))
    hits = grid.iloc[grid.sindex.query(footprint, predicate="intersects")]
    return [_tile_from_footprint(row.Name, row.geometry) for row in hits.itertuples()]
