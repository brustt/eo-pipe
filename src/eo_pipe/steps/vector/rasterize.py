from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features

from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.steps.rasterize")


@StepRegistry.register
class RasterizeStep(StepBase):
    """Rasterize a vector file onto the grid of a reference raster.

    Parameters (passed via ``**params``):
        vector_path (str | Path): **Required.** Path to the vector file.
        value_column (str): Column whose values are burned into the output.
            Defaults to ``"value"``.
        fill (float): Background fill value.  Defaults to ``0``.
        all_touched (bool): Rasterize all pixels touched by geometries.
            Defaults to ``True``.
        output_name (str): Output file stem.  Defaults to ``"rasterized"``.
    """

    name = "rasterize"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        vector_path = Path(params["vector_path"])
        value_column: str = params.get("value_column", "value")
        fill: float = float(params.get("fill", 0))
        all_touched: bool = bool(params.get("all_touched", True))
        output_name: str = params.get("output_name", "rasterized")

        gdf = gpd.read_file(vector_path)

        outputs = []
        for inp in inputs:
            out = output_dir / f"{output_name}_{inp.stem}.tif"

            with rio.open(inp) as src:
                meta = src.meta.copy()
                meta.update({"count": 1, "driver": "GTiff", "dtype": np.float32})
                transform = src.transform
                shape = (src.height, src.width)

                # Reproject vector to raster CRS if needed
                if gdf.crs != src.crs:
                    gdf_proj = gdf.to_crs(src.crs)
                else:
                    gdf_proj = gdf

                geom_values = (
                    (geom, value)
                    for geom, value in zip(gdf_proj.geometry, gdf_proj[value_column])
                )
                burned = rio.features.rasterize(
                    shapes=list(geom_values),
                    out_shape=shape,
                    transform=transform,
                    all_touched=all_touched,
                    fill=fill,
                    dtype=np.float32,
                )

            self._writer.write(out, burned[np.newaxis, :, :], **meta)

            logger.info(f"Rasterized {vector_path.name} → {out}")
            outputs.append(out)

        return StepResult(outputs=outputs)
