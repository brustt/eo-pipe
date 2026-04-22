import dataclasses
from pathlib import Path
from typing import Any, List

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

from eo_pipe.io.output_types import FlushedOutput, RasterOutput
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
    """Merge all input rasters into a single output.

    Designed for use with :class:`~eo_pipe.pipeline.batch.MergeBatch` so
    that the step receives all files at once.

    Parameters (passed via ``**params``):
        method (str): Merge resampling method: ``nearest``, ``bilinear``,
            or ``average``.  Defaults to ``"average"``.
        output_name (str): Stem of the output filename.  Defaults to
            ``"merged"``.
        to_cog (bool): Convert output to Cloud-Optimised GeoTIFF.
            Defaults to ``True``.
        nodata (float | None): Override nodata value.  If ``None``, the
            nodata from the first input is used.
    """

    name = "merge_raster"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = NamedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        if len(inputs) == 1:
            return StepOutput(outputs=[FlushedOutput(inputs[0])])

        method = params.get("method", "average")
        output_name = params.get("output_name", "merged")
        to_cog = params.get("to_cog", True)

        output_path = self._path_strategy.resolve(output_name, inputs[0], 0, output_dir)

        # Derive a merge-specific writer from the composition writer:
        # BigTIFF is always appropriate for merges; COG honours the step param.
        writer = dataclasses.replace(self._writer, bigtiff="yes", cog=to_cog)

        with rio.open(inputs[0]) as src:
            out_meta = src.meta.copy()
            nodata = params.get("nodata", src.nodata)

        if len(inputs) < 5:
            # maybe better to deal with size instead of number of files
            logger.info(f"Merging {len(inputs)} rasters via rasterio")
            src_files = [rio.open(str(p)) for p in inputs]
            try:
                merged, out_transform = merge(
                    src_files, resampling=_RESAMPLE_MAP.get(method, Resampling.average)
                )
                out_meta.update(
                    {
                        "height": merged.shape[1],
                        "width": merged.shape[2],
                        "transform": out_transform,
                    }
                )
            finally:
                for s in src_files:
                    s.close()

            return StepOutput(outputs=[RasterOutput(
                data=merged, path=output_path, meta=out_meta, writer=writer
            )])
        else:
            from osgeo_utils.gdal_merge import gdal_merge  # lazy: requires system GDAL
            logger.info(f"Merging {len(inputs)} rasters via gdal_merge")
            args = [
                "gdal_merge.py",
                "-o", str(output_path),
                "-of", "GTiff",
                "-co", "BIGTIFF=YES",
                "-co", "COMPRESS=DEFLATE",
                "-n", str(nodata),
                "-a_nodata", str(nodata),
            ] + [str(p) for p in inputs]
            gdal_merge(args)

            if to_cog:
                writer._to_cog(output_path, nodata)

            return StepOutput(outputs=[FlushedOutput(output_path)])
