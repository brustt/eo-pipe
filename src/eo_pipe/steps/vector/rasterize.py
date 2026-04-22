from pathlib import Path
from typing import Any, List

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features

from eo_pipe.io.output_types import RasterOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepOutput
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
        output_name (str): Output file stem prefix.  Defaults to
            ``"rasterized"``.
    """

    name = "rasterize"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        vector_path = Path(params["vector_path"])
        value_column: str = params.get("value_column", "value")
        fill: float = float(params.get("fill", 0))
        all_touched: bool = bool(params.get("all_touched", True))
        output_name: str = params.get("output_name", "rasterized")

        gdf = gpd.read_file(vector_path)

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(output_name, inp, 0, output_dir)

            with rio.open(inp) as src:
                meta = src.meta.copy()
                meta.update({"count": 1, "driver": "GTiff", "dtype": np.float32})
                transform = src.transform
                shape = (src.height, src.width)

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

            logger.info(f"Rasterized {vector_path.name} → {out}")
            outputs.append(RasterOutput(
                data=burned[np.newaxis, :, :],
                path=out,
                meta=meta,
                writer=self._writer,
            ))

        return StepOutput(outputs=outputs)
