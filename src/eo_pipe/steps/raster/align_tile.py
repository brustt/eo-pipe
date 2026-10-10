"""Align rasters to the Sentinel-2 Tile grid — one Aligned raster per (Tile, input)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import rasterio as rio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from eo_pipe.io.aligned import (
    DEFAULT_DATE_PATTERN,
    POLARIZATIONS,
    TAG_DATETIME,
    TAG_PARAMS,
    TAG_POLARIZATION,
    TAG_SOURCE,
    TAG_TILE,
    datetime_token,
    polarization_from_stem,
    token_to_iso,
)
from eo_pipe.io.lattice import Bounds, Lattice, intersect
from eo_pipe.io.output_types import RasterOutput
from eo_pipe.io.path_utils import TilePathStrategy
from eo_pipe.io.tiles import S2_TILE_GRID_PATH, TILE_SIZE_M, Tile, find_tiles
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.raster.resample import _RESAMPLE_METHODS

logger = setup_logger("eo_pipe.steps.raster.align_tile")


def _aoi_geometry(aoi: Union[BaseGeometry, Sequence[float]]) -> BaseGeometry:
    return aoi if isinstance(aoi, BaseGeometry) else box(*aoi)


def _valid_mask(data: np.ndarray, nodata: float) -> np.ndarray:
    if isinstance(nodata, float) and math.isnan(nodata):
        return ~np.isnan(data).all(axis=0)
    return (data != nodata).any(axis=0)


def _window_copy(src: rio.DatasetReader, dest: np.ndarray, dest_lattice: Lattice) -> None:
    """Place *src* into *dest* by whole-pixel offsets — no resampling.

    *dest_lattice* is anchored at the upper-left corner of *dest*.
    """
    col0, row0 = dest_lattice.offset(src.transform.c, src.transform.f)
    d_c0, d_c1 = max(0, col0), min(dest.shape[2], col0 + src.width)
    d_r0, d_r1 = max(0, row0), min(dest.shape[1], row0 + src.height)
    if d_c0 >= d_c1 or d_r0 >= d_r1:
        return
    window = rio.windows.Window(d_c0 - col0, d_r0 - row0, d_c1 - d_c0, d_r1 - d_r0)
    dest[:, d_r0:d_r1, d_c0:d_c1] = src.read(window=window)


@StepRegistry.register
class AlignToTileStep(StepBase):
    """Align a raster onto the pixel lattice of every Sentinel-2 Tile it touches.

    Writes one Aligned raster per (Tile, input) in the Tile's own UTM CRS, with
    the pixel edges on whole multiples of ``resolution`` from the Tile origin, so
    outputs of different dates of the same Tile differ by whole pixels and stack
    without resampling.  Inputs already on the lattice are windowed, not resampled.

    Parameters:
        resolution (float): Pixel size in metres; must divide 109 800. Default ``10.0``.
        method (str): Resampling method when a warp is needed. Default ``"bilinear"``.
        aoi (BaseGeometry | tuple | None): Extent shared by all dates, as a geometry
            or ``(minx, miny, maxx, maxy)``. Output extent = Tile ∩ AOI snapped outward.
            Without it, Tile ∩ the input bounds is used.
        aoi_crs (str): CRS of ``aoi``. Default ``"EPSG:4326"``.
        full_tile (bool): Write the whole Tile. Conflicts with ``aoi``. Default ``False``.
        min_valid_fraction (float): Skip a Tile whose window has a lower fraction of
            valid pixels; windows without any valid pixel are always skipped. Default ``0.0``.
        nodata (float | None): Output nodata. Default: the input's, else ``0``.
        grid_path (Path): Tile grid parquet. Default: the worldwide Sentinel-2 land grid.
        date_pattern (str): Regex extracting the acquisition date from the input stem.
        polarization (str | None): Overrides the detected polarization tag.
    """

    name = "align_to_tile"

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        res = float(params.get("resolution", 10.0))
        if TILE_SIZE_M % res != 0:
            raise ValueError(f"resolution {res} must divide the Tile size {TILE_SIZE_M:g} m.")
        method = params.get("method", "bilinear")
        if method not in _RESAMPLE_METHODS:
            raise ValueError(f"Unknown resampling method '{method}'. Available: {list(_RESAMPLE_METHODS)}")
        aoi = params.get("aoi")
        full_tile = bool(params.get("full_tile", False))
        if aoi is not None and full_tile:
            raise ValueError("aoi and full_tile are mutually exclusive: both define the output extent.")
        if aoi is not None:
            aoi = _aoi_geometry(aoi)
        aoi_crs = params.get("aoi_crs", "EPSG:4326")
        min_valid = float(params.get("min_valid_fraction", 0.0))
        grid_path = Path(params.get("grid_path", S2_TILE_GRID_PATH))
        date_pattern = params.get("date_pattern", DEFAULT_DATE_PATTERN)

        outputs: List[RasterOutput] = []
        for inp in inputs:
            token = datetime_token(inp.stem, date_pattern)
            with rio.open(inp) as src:
                if src.crs is None:
                    raise ValueError(f"{inp.name} has no CRS; cannot align.")
                footprint = box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
                for tile in find_tiles(footprint, grid_path):
                    out = self._align(
                        src, inp, tile, token, output_dir, res, method, aoi, aoi_crs,
                        full_tile, min_valid, params,
                    )
                    if out is not None:
                        outputs.append(out)
        return StepOutput(outputs=outputs)

    def _extent(
        self, src: rio.DatasetReader, tile: Tile, res: float,
        aoi: Optional[BaseGeometry], aoi_crs: str, full_tile: bool,
    ) -> Optional[Bounds]:
        if full_tile:
            return tile.bounds
        if aoi is not None:
            to_tile = Transformer.from_crs(aoi_crs, tile.crs, always_xy=True).transform
            target: Bounds = shp_transform(to_tile, aoi).bounds
        else:
            target = transform_bounds(src.crs, tile.crs, *src.bounds, densify_pts=21)
        clipped = intersect(tile.bounds, target)
        return Lattice.of_tile(tile, res).snap_outward(clipped) if clipped else None

    def _align(
        self, src: rio.DatasetReader, inp: Path, tile: Tile, token: str, output_dir: Path,
        res: float, method: str, aoi: Optional[BaseGeometry], aoi_crs: str,
        full_tile: bool, min_valid: float, params: Any,
    ) -> Optional[RasterOutput]:
        extent = self._extent(src, tile, res, aoi, aoi_crs, full_tile)
        if extent is None:
            return None
        x0, _, _, y1 = extent
        lattice = Lattice.of_tile(tile, res).anchored_at(extent)
        width, height = lattice.shape(extent)

        nodata = params.get("nodata")
        if nodata is None:
            nodata = src.nodata if src.nodata is not None else 0
        dest = np.full((src.count, height, width), nodata, dtype=src.dtypes[0])
        transform = from_origin(x0, y1, res, res)

        fast = src.crs == tile.crs and lattice.holds_transform(src.transform)
        if fast:
            _window_copy(src, dest, lattice)
        else:
            for band in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, band), destination=dest[band - 1],
                    src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                    dst_transform=transform, dst_crs=tile.crs, dst_nodata=nodata,
                    resampling=_RESAMPLE_METHODS[method], init_dest_nodata=True,
                )

        n_valid = int(_valid_mask(dest, nodata).sum())
        if n_valid == 0 or n_valid / dest[0].size < min_valid:
            logger.info("%s: tile %s skipped (%d valid pixels)", inp.name, tile.name, n_valid)
            return None

        tags = dict(src.tags())
        tags.update({
            TAG_TILE: tile.name,
            TAG_DATETIME: token_to_iso(token),
            TAG_SOURCE: inp.stem,
            TAG_PARAMS: json.dumps({
                "resolution": res, "method": method, "aoi": params.get("aoi") is not None,
                "full_tile": full_tile, "resampled": not fast,
            }),
        })
        pol = self._polarization(src, inp, params.get("polarization"))
        if pol:
            tags[TAG_POLARIZATION] = pol

        path = TilePathStrategy(tile.name, token).resolve(self.name, inp, 0, output_dir)
        meta = dict(
            dtype=src.dtypes[0], count=src.count, width=width, height=height,
            crs=tile.crs, transform=transform, nodata=nodata,
        )
        return RasterOutput(
            data=dest, path=path, meta=meta, writer=self._writer, tags=tags,
            descriptions=src.descriptions if any(src.descriptions) else None,
        )

    @staticmethod
    def _polarization(src: rio.DatasetReader, inp: Path, override: Optional[str]) -> Optional[str]:
        if override:
            return override.upper()
        if src.count != 1:
            return None
        desc = (src.descriptions[0] or "").upper()
        return desc if desc in POLARIZATIONS else polarization_from_stem(inp.stem)
