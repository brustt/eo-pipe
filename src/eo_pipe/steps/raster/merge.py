import dataclasses
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List

import numpy as np
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import Window

from eo_pipe.io.memory import budget_mb, plan_rows
from eo_pipe.io.output_types import FlushedOutput, WindowedRasterOutput
from eo_pipe.io.path_utils import NamedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.steps.merge_raster")

_RESAMPLE_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "average": Resampling.average,
}


@StepRegistry.register
class MergeRasterStep(StepBase):
    """Merge all input rasters into a single output, streamed in row strips.

    Designed for use with :class:`~eo_pipe.pipeline.batch.MergeBatch` so
    that the step receives all files at once.  The mosaic is never held in
    memory: it is produced and written full-width strip by strip.  Each strip
    opens only the inputs that intersect it, and only for that strip, so the
    merged size and the number of inputs are bounded by disk, not RAM, and
    open file handles by the inputs one strip spans.  Where inputs
    overlap, the first input wins.  Output grid and resolution follow the
    first input; inputs must share its CRS and dtype.  Needs rasterio only.

    Parameters (passed via ``**params``):
        method (str): Resampling used when an input's resolution differs
            from the output: ``nearest``, ``bilinear``, or ``average``.
            Defaults to ``"average"``.
        output_name (str): Stem of the output filename.  Defaults to
            ``"merged"``.
        to_cog (bool): Convert output to Cloud-Optimised GeoTIFF.
            Defaults to ``True``.
        nodata (float | None): Override nodata value.  If ``None``, the
            nodata from the first input is used.
        memory_budget_mb (float): Memory budget bounding the strip height.
            Defaults to :data:`~eo_pipe.io.memory.DEFAULT_MEMORY_BUDGET_MB`.
    """

    name = "merge_raster"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = NamedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        if len(inputs) == 1:
            return StepOutput(outputs=[FlushedOutput(inputs[0])])

        resampling = _RESAMPLE_MAP.get(params.get("method", "average"), Resampling.average)
        output_name = params.get("output_name", "merged")
        to_cog = params.get("to_cog", True)
        budget = budget_mb(params)

        output_path = self._path_strategy.resolve(output_name, inputs[0], 0, output_dir)
        # BigTIFF is always appropriate for merges; COG honours the step param.
        writer = dataclasses.replace(self._writer, bigtiff="yes", cog=to_cog)

        with rio.open(inputs[0]) as first:
            crs = first.crs
            xres, yres = first.res
            nodata = params.get("nodata", first.nodata)
            count, dtype = first.count, np.dtype(first.dtypes[0])
        boxes = _read_bounds(inputs)
        left, bottom = min(b.left for b in boxes), min(b.bottom for b in boxes)
        right, top = max(b.right for b in boxes), max(b.top for b in boxes)
        width = int(round((right - left) / xres))
        height = int(round((top - bottom) / yres))

        rows = plan_rows(budget, count * width * dtype.itemsize)
        if rows >= writer.blockysize:
            rows -= rows % writer.blockysize
        windows = [Window(0, r, width, min(rows, height - r)) for r in range(0, height, rows)]
        logger.info(
            f"Merging {len(inputs)} rasters ({width}x{height} px) in {len(windows)} strip(s) of up to {rows} rows"
        )

        fill = nodata if nodata is not None else 0

        @contextmanager
        def open_producer() -> Iterator[Callable[[Window], np.ndarray]]:
            def produce(window: Window) -> np.ndarray:
                south = top - (window.row_off + window.height) * yres
                north = top - window.row_off * yres
                # Open only the inputs this strip touches (first wins, so keep input order).
                hit = [p for p, b in zip(inputs, boxes) if b.bottom < north and b.top > south]
                if not hit:
                    return np.full((count, int(window.height), width), fill, dtype=dtype)
                with ExitStack() as stack:
                    sources = [stack.enter_context(rio.open(p)) for p in hit]
                    data, _ = merge(
                        sources, bounds=(left, south, right, north), res=(xres, yres), nodata=nodata,
                        dtype=dtype.name, resampling=resampling, method="first",
                    )
                return data[:, : int(window.height), : int(window.width)]

            yield produce

        meta = {
            "crs": crs, "dtype": dtype.name, "count": count, "width": width, "height": height,
            "transform": from_origin(left, top, xres, yres), "nodata": nodata,
        }
        return StepOutput(outputs=[WindowedRasterOutput(
            windows=windows, open_producer=open_producer, path=output_path, meta=meta, writer=writer,
        )])


def _read_bounds(paths: List[Path]) -> List[BoundingBox]:
    """Bounds of every raster in *paths*, in order."""
    boxes = []
    for path in paths:
        with rio.open(path) as src:
            boxes.append(src.bounds)
    return boxes
