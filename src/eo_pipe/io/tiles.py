"""Sentinel-2 MGRS Tile lookup.

This module also owns UTM zone arithmetic (:func:`utm_epsg`, :func:`utm_zone`) so
that no step derives a zone on its own.

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
from typing import List, Optional, Tuple, Union

import geopandas as gpd
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from eo_pipe.io.vector_io import _read_geodataframe

#: Default Sentinel-2 Tile grid (ocean-masked, EPSG:4326 footprints, ``Name`` column).
S2_TILE_GRID_PATH = Path("/home/brustt/Documents/Projects/data/s2_mgrs_worldwide_grid_land.parquet")

#: Side of a Sentinel-2 Tile in metres.
TILE_SIZE_M = 109_800.0

#: Tile corners ``(UL, UR, LR, LL)``, each ``(lon, lat)`` in WGS 84.
CxPolygon = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

#: WGS 84 / NSIDC EASE-Grid 2.0 Global (equal-area, metres) — for ranking overlaps.
_EQUAL_AREA_EPSG = 6933
_TILE_NAME_COLUMNS = ("Name", "tile_id", "TILE_ID", "name")

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

    @property
    def corners_wgs84(self) -> CxPolygon:
        """Tile extent as ``(UL, UR, LR, LL)`` corners, ``(lon, lat)`` in WGS 84."""
        minx, miny, maxx, maxy = transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21)
        return ((minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny))


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing ``(lon, lat)`` (WGS 84 degrees)."""
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def utm_zone(epsg: int) -> Optional[Tuple[int, bool]]:
    """``(zone, is_north)`` for a WGS 84 UTM EPSG code, ``None`` for any other code."""
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    return None


def tile_epsg(name: str) -> int:
    """EPSG code of the UTM zone of tile *name* (zone and hemisphere from the name)."""
    zone = int(name[:2])
    base = 32700 if name[2].upper() in _SOUTHERN_BANDS else 32600
    return base + zone


@lru_cache(maxsize=4)
def _load_grid(path: str) -> gpd.GeoDataFrame:
    """Tile grid in EPSG:4326 with the tile name in a ``Name`` column."""
    gdf = _read_geodataframe(Path(path))
    column = next((c for c in _TILE_NAME_COLUMNS if c in gdf.columns), None)
    if column is None:
        raise ValueError(
            f"Tile grid {path} has no tile-name column; expected one of {_TILE_NAME_COLUMNS}."
        )
    if column != "Name":
        gdf = gdf.rename(columns={column: "Name"})
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


def best_tile(aoi: Union[str, Path], grid_path: Union[str, Path] = S2_TILE_GRID_PATH) -> Tile:
    """The Tile overlapping the AOI file the most (compared in an equal-area CRS).

    Raises:
        ValueError: If no Tile intersects the AOI.
    """
    aoi_geom = _read_geodataframe(Path(aoi)).to_crs(4326).union_all()
    grid = _load_grid(str(grid_path))
    candidates = grid.iloc[grid.sindex.query(aoi_geom, predicate="intersects")]
    if candidates.empty:
        raise ValueError(f"No Sentinel-2 Tile intersects the study area: {aoi}")
    to_ea = Transformer.from_crs(4326, _EQUAL_AREA_EPSG, always_xy=True).transform
    aoi_ea = shp_transform(to_ea, aoi_geom)
    overlaps = [shp_transform(to_ea, g).intersection(aoi_ea).area for g in candidates.geometry]
    best = candidates.iloc[max(range(len(overlaps)), key=overlaps.__getitem__)]
    return _tile_from_footprint(best.Name, best.geometry)
