from pathlib import Path
from typing import Any, List, Union

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

logger = setup_logger("eo_pipe.steps.hillshade")


def create_hillshade(
    dem_path: Union[str, Path],
    output_path: Union[str, Path],
    z_factor: float = 1.0,
    azimuth: float = 300.0,
    altitude: float = 45.0,
) -> Path:
    """Create a hillshade from a DEM raster using GDAL DEMProcessing.

    Requires system GDAL (``osgeo``).

    Args:
        dem_path: Input DEM raster path.
        output_path: Destination path for the hillshade.
        z_factor: Vertical exaggeration factor.
        azimuth: Light source azimuth angle in degrees.
        altitude: Light source altitude angle in degrees.

    Returns:
        Resolved output path.

    Raises:
        IOError: If the DEM file cannot be opened.
    """
    from osgeo import gdal  # lazy: osgeo requires system GDAL

    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dem_dataset = gdal.Open(str(dem_path))
    if not dem_dataset:
        raise IOError(f"Could not open DEM file: {dem_path}")

    options = gdal.DEMProcessingOptions(
        format="GTiff",
        zFactor=z_factor,
        azimuth=azimuth,
        altitude=altitude,
    )
    gdal.DEMProcessing(
        destName=str(output_path),
        srcDS=dem_dataset,
        processing="hillshade",
        options=options,
    )
    dem_dataset = None  # Close GDAL dataset

    logger.info(f"Hillshade written to {output_path}")
    return output_path


@StepRegistry.register
class HillshadeStep(StepBase):
    """Generate a hillshade from DEM rasters using GDAL DEMProcessing.

    Requires system GDAL (``osgeo``).  Tests auto-skip when ``osgeo`` is
    unavailable (same pattern as :class:`~eo_pipe.steps.raster.sieve.SieveStep`).

    Parameters (passed via ``**params``):
        z_factor (float): Vertical exaggeration factor.  Defaults to ``1.0``.
        azimuth (float): Light source azimuth in degrees.  Defaults to ``300.0``.
        altitude (float): Light source altitude in degrees.  Defaults to ``45.0``.
    """

    name = "hillshade"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        z_factor: float = float(params.get("z_factor", 1.0))
        azimuth: float = float(params.get("azimuth", 300.0))
        altitude: float = float(params.get("altitude", 45.0))

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            create_hillshade(inp, out, z_factor=z_factor, azimuth=azimuth, altitude=altitude)
            outputs.append(FlushedOutput(out))

        return StepOutput(outputs=outputs)
