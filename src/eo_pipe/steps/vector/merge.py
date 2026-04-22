from pathlib import Path
from typing import Any, List

import geopandas as gpd
import pandas as pd

from eo_pipe.io.output_types import GpkgFormat, VectorFormat, VectorOutput
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.steps.merge_vector")


@StepRegistry.register
class MergeVectorStep(StepBase):
    """Merge all input vector files into a single output.

    Designed for use with :class:`~eo_pipe.pipeline.batch.MergeBatch`.

    Parameters (passed via ``**params``):
        output_name (str): Stem of the output file.  Defaults to ``"merged"``.
        fmt (:class:`~eo_pipe.io.output_types.VectorFormat`): Format strategy
            controlling the file extension and write implementation.
            Defaults to :class:`~eo_pipe.io.output_types.GpkgFormat`.
        dissolve (bool): If ``True``, attempt a unary union of all geometries
            before saving.  Defaults to ``False``.

    Example::

        from eo_pipe.io.output_types import ParquetFormat

        pipeline.add_step(
            "merge_vector",
            MergeBatch(),
            output_name="combined",
            fmt=ParquetFormat(),
        )
    """

    name = "merge_vector"

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        output_name = params.get("output_name", "merged")
        fmt: VectorFormat = params.get("fmt", GpkgFormat())
        dissolve = bool(params.get("dissolve", False))

        output_path = output_dir / f"{output_name}{fmt.extension}"

        gdfs = []
        for p in inputs:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Vector file not found: {p}")
            try:
                gdf = gpd.read_file(p)
                if not gdf.empty:
                    gdfs.append(gdf)
            except Exception as e:
                logger.warning(f"Could not read {p}: {e}")

        if not gdfs:
            raise ValueError("No valid vector files to merge.")

        combined = pd.concat(gdfs, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, crs=gdfs[0].crs)

        if dissolve:
            try:
                union_geom = combined.geometry.union_all()
                combined = gpd.GeoDataFrame(
                    {"geometry": [union_geom]}, crs=gdfs[0].crs
                )
                logger.info("Dissolved all geometries into a single feature.")
            except Exception as e:
                logger.warning(f"Dissolve failed, keeping concatenated result: {e}")

        return StepOutput(outputs=[VectorOutput(gdf=combined, path=output_path, fmt=fmt)])
