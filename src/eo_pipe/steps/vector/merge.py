from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd

from eo_pipe.io.vector_io import save_vector
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.steps.merge_vector")


@StepRegistry.register
class MergeVectorStep(StepBase):
    """Merge all input vector files into a single output.

    Designed for use with :class:`~eo_pipe.pipeline.batch.MergeBatch`.

    Parameters (passed via ``**params``):
        output_name (str): Stem of the output file.  Defaults to ``"merged"``.
        driver (str): OGR driver.  Defaults to ``"ESRI Shapefile"``.
        dissolve (bool): If ``True``, attempt a unary union of all geometries
            before saving.  Defaults to ``False``.
    """

    name = "merge_vector"

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        output_name = params.get("output_name", "merged")
        driver = params.get("driver", "ESRI Shapefile")
        dissolve = bool(params.get("dissolve", False))

        ext = ".gpkg" if driver == "GPKG" else ".shp"
        output_path = output_dir / f"{output_name}{ext}"

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
                union_geom = combined.geometry.unary_union
                combined = gpd.GeoDataFrame(
                    {"geometry": [union_geom]}, crs=gdfs[0].crs
                )
                logger.info("Dissolved all geometries into a single feature.")
            except Exception as e:
                logger.warning(f"Dissolve failed, keeping concatenated result: {e}")

        save_vector(combined, output_path, driver=driver)
        return StepResult(outputs=[output_path])
