from pathlib import Path
from typing import List

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

from eo_pipe.io.raster_io import RasterWriter
from eo_pipe.pipeline.base import StepBase, StepResult
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

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        if len(inputs) == 1:
            return StepResult(outputs=list(inputs))

        method = params.get("method", "average")
        output_name = params.get("output_name", "merged")
        to_cog = params.get("to_cog", True)

        output_path = output_dir / f"{output_name}.tif"

        writer = RasterWriter(
            compress="deflate",
            bigtiff="yes",
            cog=to_cog,
        )

        with rio.open(inputs[0]) as src:
            out_meta = src.meta.copy()
            nodata = params.get("nodata", src.nodata)

        if len(inputs) < 100:
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
                writer.write(output_file=output_path, data=merged, **out_meta)
            finally:
                for s in src_files:
                    s.close()
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

        return StepResult(outputs=[output_path])
