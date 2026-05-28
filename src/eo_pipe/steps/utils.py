from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import osgeo
import osgeo.osr as osr
from eo_pipe.io.vector_io import _read_geodataframe

Polygon = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

# EPSG:6933 — WGS 84 / NSIDC EASE-Grid 2.0 World (equal-area, metres)
_EQUAL_AREA_EPSG = 6933

_S2_TILE_NAME_CANDIDATES = ("Name", "tile_id", "TILE_ID", "name")


def _detect_tile_col(columns) -> str:
    for candidate in _S2_TILE_NAME_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"No tile-name column found in S2 grid. "
        f"Expected one of {_S2_TILE_NAME_CANDIDATES}, got: {list(columns)}"
    )


def get_best_s2_tile(
    study_area: Path,
    s2_grid: Path,
) -> Tuple[str, Polygon]:
    """Return the S2 MGRS tile that overlaps *study_area* the most.

    Parameters
    ----------
    study_area:
        Path to a ``.gpkg`` or ``.parquet`` vector file defining the area of interest.
    s2_grid:
        Path to the Sentinel-2 MGRS tile grid (``.gpkg``, ``.parquet``, ``.shp``, …).

    Returns
    -------
    tile_name : str
        MGRS tile identifier, e.g. ``"31TCK"``.
    tile_extent : Polygon
        Tile corners in **WGS 84** (EPSG:4326) as ``(UL, UR, LR, LL)`` where
        each corner is ``(lon, lat)``.  Compatible with :func:`s2_tile_extent`.

    Raises
    ------
    ValueError
        If no S2 tile intersects the study area.
    """

    aoi = _read_geodataframe(study_area).to_crs(epsg=4326)
    grid = _read_geodataframe(s2_grid).to_crs(epsg=4326)

    tile_col = _detect_tile_col(grid.columns)

    aoi_union = aoi.union_all()
    candidates = grid[grid.intersects(aoi_union)].copy()

    if candidates.empty:
        raise ValueError(
            f"No S2 MGRS tile intersects the study area: {study_area}"
        )

    # Compute overlap in equal-area projection for correct area ranking.
    candidates_ea = candidates.to_crs(epsg=_EQUAL_AREA_EPSG)
    aoi_ea = aoi.to_crs(epsg=_EQUAL_AREA_EPSG).union_all()

    candidates["_overlap_m2"] = candidates_ea.geometry.intersection(aoi_ea).area

    best = candidates.loc[candidates["_overlap_m2"].idxmax()]
    tile_name: str = best[tile_col]
    b = best.geometry.bounds  # (minx, miny, maxx, maxy) in WGS 84
    # Build Polygon as (UL, UR, LR, LL) in (lon, lat) — matches s2_tile_extent expectation.
    tile_extent: Polygon = (
        (b.minx, b.maxy),  # UL
        (b.maxx, b.maxy),  # UR
        (b.maxx, b.miny),  # LR
        (b.minx, b.miny),  # LL
    )

    return tile_name, tile_extent


def convert_coord(
    tuple_list: List[Tuple[float, float]],
    in_epsg:    int,
    out_epsg:   int,
) -> List[Tuple[float, ...]]:
    """
    Convert a list of coordinates from one epsg code to another

    Args:
      tuple_list: a list of tuples representing the coordinates
      in_epsg: the input epsg code
      out_epsg: the output epsg code

    Returns:
      a list of tuples representing the converted coordinates
    """
    if not tuple_list:
        return []

    tuple_out = []

    in_spatial_ref = osr.SpatialReference()
    in_spatial_ref.ImportFromEPSG(in_epsg)
    out_spatial_ref = osr.SpatialReference()
    out_spatial_ref.ImportFromEPSG(out_epsg)
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 2.0 and GDAL 3.0 don't take the CoordinateTransformation() parameters
        # in the same order: https://github.com/OSGeo/gdal/issues/1546
        #
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        in_spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        # out_spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    for in_coord in tuple_list:
        lon = in_coord[0]
        lat = in_coord[1]

        coord_trans = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)
        coord = coord_trans.TransformPoint(lon, lat)
        # logger.debug("convert_coord(lon=%s, lat=%s): %s, %s ==> %s", in_epsg, out_epsg, lon, lat, coord)
        tuple_out.append(coord)
    return tuple_out