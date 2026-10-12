"""Align rasters to the Sentinel-2 Tile grid — one Aligned raster per (Tile, input)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
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


#: Source pixels needed beyond the footprint, per source pixel per destination pixel (largest kernel: lanczos).
_KERNEL_RADIUS = 3
#: Fewest rows of a window; GDAL's source-window estimate is unstable on thinner ones.
_MIN_WINDOW_ROWS = MIN_CHUNK_SIDE


@dataclass(frozen=True)
class _Options:
    """Per-run options of :class:`AlignToTileStep`, shared by every (Tile, input)."""

    resolution: float
    method: str
    aoi: Optional[BaseGeometry]
    aoi_crs: str
    full_tile: bool
    min_valid_fraction: float
    memory_budget_mb: float
    nodata: Optional[float]
    polarization: Optional[str]


@dataclass(frozen=True)
class _Target:
    """The output grid of one Aligned raster: its Tile, extent lattice and pixel format."""

    tile: Tile
    lattice: Lattice        # anchored at the upper-left of the extent
    width: int
    height: int
    nodata: float
    dtype: str
    count: int
    fast: bool              # the source already sits on the lattice: copy, do not resample

    @property
    def transform(self) -> Affine:
        return self.lattice.transform


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


def _warp_options(src: rio.DatasetReader, target: _Target) -> dict:
    """GDAL warp options that make a windowed warp equal a single-shot one.

    GDAL derives the resampling scale and the source margin per chunk from the part of the
    source a window touches, so the same pixel would resample differently depending on how
    the output is cut.  Both are pinned from the footprint of the *whole* output instead
    (never a window's, whose ratio drifts with latitude): ``XSCALE`` /
    ``YSCALE`` are destination over source pixel size (below 1 when downsampling) and
    ``SOURCE_EXTRA`` is the kernel margin in source pixels.
    """
    bounds = transform_bounds(
        target.tile.crs, src.crs, *target.lattice.bounds(target.width, target.height), densify_pts=21,
    )
    window = src.window(*bounds)
    ratio_x = max(window.width / target.width, 1.0)          # source px per destination px
    ratio_y = max(window.height / target.height, 1.0)
    return {
        "XSCALE": 1.0 / ratio_x,
        "YSCALE": 1.0 / ratio_y,
        "SOURCE_EXTRA": math.ceil(_KERNEL_RADIUS * max(ratio_x, ratio_y)) + 2,
    }


def _plan_windows(target: _Target, budget: float, block_rows: int) -> List[Window]:
    """Full-width strips sized from the *Memory budget*.

    The warp approximation is per scanline, so strip height never changes pixels.  Strips are
    tile-aligned when the budget allows, and never thinner than ``_MIN_WINDOW_ROWS``.
    """
    rows = plan_rows(budget, target.count * target.width * np.dtype(target.dtype).itemsize)
    if rows >= block_rows:
        rows -= rows % block_rows
    rows = max(rows, min(_MIN_WINDOW_ROWS, target.height))
    starts = list(range(0, target.height, rows))
    if len(starts) > 1 and target.height - starts[-1] < _MIN_WINDOW_ROWS:
        starts.pop()                          # fold a too-thin last window into the previous one
    ends = starts[1:] + [target.height]
    return [Window(0, r0, target.width, r1 - r0) for r0, r1 in zip(starts, ends)]


def _window_producer(
    src: rio.DatasetReader, target: _Target, method: str,
) -> Callable[[Window], np.ndarray]:
    """A callable mapping a window of the output to its pixels, copying or warping *src*."""
    warp_opts = {} if target.fast else _warp_options(src, target)

    def produce(window: Window) -> np.ndarray:
        dest = np.full((target.count, int(window.height), int(window.width)), target.nodata, dtype=target.dtype)
        window_lattice = target.lattice.at_window(window)
        if target.fast:
            _window_copy(src, dest, window_lattice)
            return dest
        reproject(
            source=rio.band(src, list(range(1, src.count + 1))), destination=dest,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=window_lattice.transform, dst_crs=target.tile.crs, dst_nodata=target.nodata,
            resampling=_RESAMPLE_METHODS[method], init_dest_nodata=True,
            tolerance=0.0,   # exact transform and kernel margin: output must not depend on window geometry
            **warp_opts,
        )
        return dest

    return produce


def _count_valid(
    produce: Callable[[Window], np.ndarray], windows: Sequence[Window], nodata: float, min_valid: float,
) -> int:
    """Valid pixels over *windows*, stopping at the first one when ``min_valid`` is 0.

    Accumulated window by window so a skipped Tile writes nothing.
    """
    n_valid = 0
    for window in windows:
        n_valid += int(_valid_mask(produce(window), nodata).sum())
        if min_valid <= 0 and n_valid > 0:
            break
    return n_valid


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
        opts = _Options(
            resolution=res, method=method,
            aoi=_aoi_geometry(aoi) if aoi is not None else None,
            aoi_crs=params.get("aoi_crs", "EPSG:4326"),
            full_tile=full_tile,
            min_valid_fraction=float(params.get("min_valid_fraction", 0.0)),
            memory_budget_mb=budget_mb(params),
            nodata=params.get("nodata"),
            polarization=params.get("polarization"),
        )
        grid_path = Path(params.get("grid_path", S2_TILE_GRID_PATH))
        date_pattern = params.get("date_pattern", DEFAULT_DATE_PATTERN)

        outputs: List[WindowedRasterOutput] = []
        for inp in inputs:
            token = datetime_token(inp.stem, date_pattern)
            with rio.open(inp) as src:
                if src.crs is None:
                    raise ValueError(f"{inp.name} has no CRS; cannot align.")
                footprint = box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
                for tile in find_tiles(footprint, grid_path):
                    out = self._align(src, inp, tile, token, output_dir, opts)
                    if out is not None:
                        outputs.append(out)
        return StepOutput(outputs=outputs)

    def _extent(self, src: rio.DatasetReader, tile: Tile, opts: _Options) -> Optional[Bounds]:
        """Output extent within *tile*, snapped to the lattice; ``None`` when *src* misses it."""
        if opts.full_tile:
            return tile.bounds
        if opts.aoi is not None:
            to_tile = Transformer.from_crs(opts.aoi_crs, tile.crs, always_xy=True).transform
            target: Bounds = shp_transform(to_tile, opts.aoi).bounds
        else:
            target = transform_bounds(src.crs, tile.crs, *src.bounds, densify_pts=21)
        clipped = intersect(tile.bounds, target)
        return Lattice.of_tile(tile, opts.resolution).snap_outward(clipped) if clipped else None

    def _target(self, src: rio.DatasetReader, tile: Tile, opts: _Options) -> Optional[_Target]:
        extent = self._extent(src, tile, opts)
        if extent is None:
            return None
        lattice = Lattice.of_tile(tile, opts.resolution).anchored_at(extent)
        width, height = lattice.shape(extent)
        nodata = opts.nodata
        if nodata is None:
            nodata = src.nodata if src.nodata is not None else 0
        return _Target(
            tile=tile, lattice=lattice, width=width, height=height, nodata=nodata,
            dtype=src.dtypes[0], count=src.count,
            fast=src.crs == tile.crs and lattice.holds_transform(src.transform),
        )

    def _pending_output(
        self, src: rio.DatasetReader, inp: Path, token: str, output_dir: Path,
        target: _Target, windows: List[Window], opts: _Options,
    ) -> WindowedRasterOutput:
        """The deferred write of one Aligned raster: tags, path, profile and window producer."""
        tags = tags_for(
            src.tags(), tile=target.tile.name, token=token, source=inp.stem,
            params={
                "resolution": opts.resolution, "method": opts.method, "aoi": opts.aoi is not None,
                "full_tile": opts.full_tile, "resampled": not target.fast,
            },
            polarization=polarization_of(src.count, src.descriptions, inp.stem, opts.polarization),
        )
        path = TilePathStrategy(target.tile.name, token).resolve(self.name, inp, 0, output_dir)
        meta = dict(
            dtype=target.dtype, count=target.count, width=target.width, height=target.height,
            crs=target.tile.crs, transform=target.transform, nodata=target.nodata,
        )

        @contextmanager
        def open_producer() -> Iterator[Callable[[Window], np.ndarray]]:
            with rio.open(inp) as source:
                yield _window_producer(source, target, opts.method)

        return WindowedRasterOutput(
            windows=windows, open_producer=open_producer, path=path, meta=meta,
            writer=self._writer, tags=tags,
            descriptions=src.descriptions if any(src.descriptions) else None,
        )

    def _align(
        self, src: rio.DatasetReader, inp: Path, tile: Tile, token: str, output_dir: Path, opts: _Options,
    ) -> Optional[WindowedRasterOutput]:
        target = self._target(src, tile, opts)
        if target is None:
            return None
        windows = _plan_windows(target, opts.memory_budget_mb, self._writer.blockysize)
        logger.info(
            "%s: tile %s, %dx%d px in %d window(s) of up to %d rows",
            inp.name, tile.name, target.width, target.height, len(windows), max(w.height for w in windows),
        )
        n_valid = _count_valid(_window_producer(src, target, opts.method), windows, target.nodata, opts.min_valid_fraction)
        if n_valid == 0 or n_valid / (target.width * target.height) < opts.min_valid_fraction:
            logger.info("%s: tile %s skipped (%d valid pixels)", inp.name, tile.name, n_valid)
            return None
        return self._pending_output(src, inp, token, output_dir, target, windows, opts)
