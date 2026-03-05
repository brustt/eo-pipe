from pathlib import Path
from typing import List

import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.steps.clip")


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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        shp = params["shp"]
        save_overlay = params.get("save_overlay", True)

        if isinstance(shp, (str, Path)):
            shp = gpd.read_file(shp)

        outputs = []
        artifacts = {}

        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            overlay_path = None

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

            self._writer.write(output_file=out, data=out_image, **out_meta)
            outputs.append(out)

            if save_overlay:
                overlay_path = _extract_overlay(
                    inp, shp, raster_crs, raster_bounds, output_dir
                )
                artifacts[f"overlay_{inp.stem}"] = overlay_path

        return StepResult(outputs=outputs, artifacts=artifacts)


def _extract_overlay(
    raster_path: Path,
    vector_gdf: gpd.GeoDataFrame,
    raster_crs,
    raster_bounds,
    output_dir: Path,
) -> Path:
    """Save the intersection of *vector_gdf* with the raster extent."""
    bbox = box(
        raster_bounds.left,
        raster_bounds.bottom,
        raster_bounds.right,
        raster_bounds.top,
    )
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=raster_crs)

    if vector_gdf.crs != raster_crs:
        vector_gdf = vector_gdf.to_crs(raster_crs)

    overlay_gdf = gpd.overlay(vector_gdf, bbox_gdf, how="intersection")
    overlay_path = output_dir / f"{raster_path.stem}_overlay.shp"
    overlay_gdf.to_file(overlay_path)
    logger.info(f"Vector overlay saved to {overlay_path}")
    return overlay_path
