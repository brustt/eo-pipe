from pathlib import Path
from typing import Any, List

import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

from eo_pipe.io.output_types import RasterOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry


@StepRegistry.register
class ClipStep(StepBase):
    """Clip each input raster to a vector geometry.

    Parameters (passed via ``**params``):
        shp (str | Path | GeoDataFrame): Clip geometry.  If a path, the
            file is read with geopandas.
        save_overlay (bool): If ``True`` (default), save the intersection of
            *shp* with the raster extent as a sidecar shapefile.  The path
            is stored in ``StepResult.artifacts["overlay_<stem>"]``.
    """

    name = "clip"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        shp = params["shp"]

        if isinstance(shp, (str, Path)):
            shp = gpd.read_file(shp)

        outputs: list[RasterOutput] = []
        artifacts: dict[str, Path] = {}

        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)

            with rio.open(inp) as src:
                raster_crs = src.crs
                raster_bounds = src.bounds
                shp_projected = shp.to_crs(raster_crs) if shp.crs != raster_crs else shp
                out_image, out_transform = rio_mask(
                    src, shp_projected.geometry, filled=True, crop=True
                )
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

            outputs.append(RasterOutput(data=out_image, path=out, meta=out_meta, writer=self._writer))

        return StepOutput(outputs=outputs, artifacts=artifacts)
