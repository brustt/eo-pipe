"""Generic vector I/O utilities."""

from pathlib import Path
from typing import Union

import geopandas as gpd

from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.io.vector")


def save_vector(
    gdf: gpd.GeoDataFrame,
    output_path: Union[str, Path],
    driver: str = "ESRI Shapefile",
) -> Path:
    """Write *gdf* to *output_path*.

    Args:
        gdf: GeoDataFrame to save.
        output_path: Destination path.
        driver: OGR driver name (default: ``"ESRI Shapefile"``).

    Returns:
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver=driver)
    logger.info(f"Vector saved to {output_path}")
    return output_path
