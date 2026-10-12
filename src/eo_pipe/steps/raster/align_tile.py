"""Align rasters to the Sentinel-2 Tile grid — one Aligned raster per (Tile, input)."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from eo_pipe.io.aligned import (
    DEFAULT_DATE_PATTERN,
    datetime_token,
    polarization_of,
    tags_for,
)
from eo_pipe.io.lattice import Bounds, Lattice, intersect
from eo_pipe.io.memory import MIN_CHUNK_SIDE, budget_mb, plan_rows
from eo_pipe.io.output_types import WindowedRasterOutput
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


#: Source pixels needed beyond the footprint, per source pixel per destination pixel (largest kernel: lanczos).
_KERNEL_RADIUS = 3
#: Fewest rows of a window; GDAL's source-window estimate is unstable on thinner ones.
_MIN_WINDOW_ROWS = MIN_CHUNK_SIDE


def _warp_options(
    src: rio.DatasetReader, dst_transform: Affine, width: int, height: int, dst_crs: CRS, dst_res: float,
) -> dict:
    """GDAL warp options that make a windowed warp equal a single-shot one.

    GDAL derives the resampling scale and the source margin per chunk from the part of the
    source a window touches, so the same pixel would resample differently depending on how
    the output is cut.  Both are pinned from the footprint of the *whole* output instead
    (never a window's, whose ratio drifts with latitude): ``XSCALE`` /
    ``YSCALE`` are destination over source pixel size (below 1 when downsampling) and
    ``SOURCE_EXTRA`` is the kernel margin in source pixels.
    """
    left, top = dst_transform.c, dst_transform.f
    bounds = transform_bounds(
        dst_crs, src.crs, left, top - height * dst_res, left + width * dst_res, top, densify_pts=21,
    )
    window = src.window(*bounds)
    ratio_x = max(window.width / width, 1.0)          # source px per destination px
    ratio_y = max(window.height / height, 1.0)
    return {
        "XSCALE": 1.0 / ratio_x,
        "YSCALE": 1.0 / ratio_y,
        "SOURCE_EXTRA": math.ceil(_KERNEL_RADIUS * max(ratio_x, ratio_y)) + 2,
    }


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
        memory_budget_mb (float): Ceiling for the step, in MB. Each Aligned raster is
            produced and written in full-width strips sized from it, one Tile at a time,
            so no full-Tile array is ever held. Output is independent of the budget.
            Raises if not even a minimal window fits. Default ``4096``.
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

        budget = budget_mb(params)
        outputs: List[WindowedRasterOutput] = []
        for inp in inputs:
            token = datetime_token(inp.stem, date_pattern)
            with rio.open(inp) as src:
                if src.crs is None:
                    raise ValueError(f"{inp.name} has no CRS; cannot align.")
                footprint = box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
                for tile in find_tiles(footprint, grid_path):
                    out = self._align(
                        src, inp, tile, token, output_dir, res, method, aoi, aoi_crs,
                        full_tile, min_valid, budget, params,
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
        full_tile: bool, min_valid: float, budget: float, params: Any,
    ) -> Optional[WindowedRasterOutput]:
        extent = self._extent(src, tile, res, aoi, aoi_crs, full_tile)
        if extent is None:
            return None
        x0, _, _, y1 = extent
        lattice = Lattice.of_tile(tile, res).anchored_at(extent)
        width, height = lattice.shape(extent)

        nodata = params.get("nodata")
        if nodata is None:
            nodata = src.nodata if src.nodata is not None else 0
        dtype = src.dtypes[0]
        transform = from_origin(x0, y1, res, res)
        fast = src.crs == tile.crs and lattice.holds_transform(src.transform)

        # Full-width strips: the warp approximation is per scanline, so strip height never changes pixels.
        rows = plan_rows(budget, src.count * width * np.dtype(dtype).itemsize)
        if rows >= self._writer.blockysize:
            rows -= rows % self._writer.blockysize
        rows = max(rows, min(_MIN_WINDOW_ROWS, height))
        starts = list(range(0, height, rows))
        if len(starts) > 1 and height - starts[-1] < _MIN_WINDOW_ROWS:
            starts.pop()                          # fold a too-thin last window into the previous one
        bounds_ = starts[1:] + [height]
        windows = [Window(0, r0, width, r1 - r0) for r0, r1 in zip(starts, bounds_)]
        logger.info(
            "%s: tile %s, %dx%d px in %d window(s) of up to %d rows",
            inp.name, tile.name, width, height, len(windows), rows,
        )

        warp_opts = {} if fast else _warp_options(src, transform, width, height, tile.crs, res)

        def window_fn(src: rio.DatasetReader) -> Callable[[Window], np.ndarray]:
            def produce(window: Window) -> np.ndarray:
                r0, h = int(window.row_off), int(window.height)
                dest = np.full((src.count, h, width), nodata, dtype=dtype)
                if fast:
                    _window_copy(src, dest, Lattice(x0, y1 - r0 * res, res))
                    return dest
                dst_transform = transform * Affine.translation(0, r0)
                reproject(
                    source=rio.band(src, list(range(1, src.count + 1))), destination=dest,
                    src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                    dst_transform=dst_transform, dst_crs=tile.crs, dst_nodata=nodata,
                    resampling=_RESAMPLE_METHODS[method], init_dest_nodata=True,
                    tolerance=0.0,   # exact transform and kernel margin: output must not depend on window geometry
                    **warp_opts,
                )
                return dest
            return produce

        # Valid-pixel count is accumulated window by window, so a skipped Tile writes nothing.
        produce = window_fn(src)
        n_valid = 0
        for window in windows:
            n_valid += int(_valid_mask(produce(window), nodata).sum())
            if min_valid <= 0 and n_valid > 0:
                break
        if n_valid == 0 or n_valid / (width * height) < min_valid:
            logger.info("%s: tile %s skipped (%d valid pixels)", inp.name, tile.name, n_valid)
            return None

        tags = tags_for(
            src.tags(), tile=tile.name, token=token, source=inp.stem,
            params={
                "resolution": res, "method": method, "aoi": params.get("aoi") is not None,
                "full_tile": full_tile, "resampled": not fast,
            },
            polarization=polarization_of(src.count, src.descriptions, inp.stem, params.get("polarization")),
        )

        path = TilePathStrategy(tile.name, token).resolve(self.name, inp, 0, output_dir)
        meta = dict(
            dtype=src.dtypes[0], count=src.count, width=width, height=height,
            crs=tile.crs, transform=transform, nodata=nodata,
        )

        @contextmanager
        def open_producer() -> Iterator[Callable[[Window], np.ndarray]]:
            with rio.open(inp) as source:
                yield window_fn(source)

        return WindowedRasterOutput(
            windows=windows, open_producer=open_producer, path=path, meta=meta,
            writer=self._writer, tags=tags,
            descriptions=src.descriptions if any(src.descriptions) else None,
        )
