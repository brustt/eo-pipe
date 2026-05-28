"""Sentinel-1 geocoding step — warps GCP-tagged rasters to a projected CRS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import rasterio

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

logger = setup_logger("eo_pipe.steps.raster.s1_geocode")


def _auto_utm_crs(inp: Path) -> str:
    """Derive UTM zone CRS from the GCP centroid of *inp*.

    Falls back to EPSG:4326 when no GCPs are present.
    The GCP CRS (always geographic) is intentionally not used as the target —
    10m resolution in degrees would produce a near-empty output.
    """
    with rasterio.open(inp) as src:
        gcps, _ = src.gcps
    if not gcps:
        return "EPSG:4326"
    centre_lon = sum(g.x for g in gcps) / len(gcps)
    centre_lat = sum(g.y for g in gcps) / len(gcps)
    zone = int((centre_lon + 180) / 6) + 1
    epsg = 32600 + zone if centre_lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


@StepRegistry.register
class S1GeocodeStep(StepBase):
    """Geocode a GCP-tagged Sentinel-1 raster to a projected CRS via GDAL Warp (TPS).

    Accepts GeoTIFF outputs of :class:`S1ExtractStep` (GCPs from the annotation
    XML embedded) and reprojects to a target CRS using Thin Plate Spline
    transformation.

    Parameters:
        dst_crs (str | None): Target CRS as EPSG string, e.g. ``"EPSG:32632"``.
            When omitted, auto-detected from the UTM zone of the GCP centroid.
        x_res (float): Output pixel width in target CRS units. Default ``10.0``.
        y_res (float): Output pixel height in target CRS units. Default ``10.0``.
        resample_alg (str): GDAL resampling algorithm. Default ``"bilinear"``.
        src_nodata (float): Source nodata value. Default ``0``.
        dst_nodata (float): Destination nodata value. Default ``0``.
    """

    name = "s1_geocode"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def is_available(self) -> bool:
        try:
            from osgeo import gdal  # noqa: F401
            return True
        except ImportError:
            return False

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        dst_crs: Optional[str] = params.get("dst_crs", None)
        x_res: float = float(params.get("x_res", 10.0))
        y_res: float = float(params.get("y_res", 10.0))
        resample_alg: str = params.get("resample_alg", "bilinear")
        src_nodata: float = float(params.get("src_nodata", 0))
        dst_nodata: float = float(params.get("dst_nodata", 0))

        outputs: list[FlushedOutput] = []
        for inp in inputs:
            crs = dst_crs or _auto_utm_crs(inp)
            logger.info("Auto-detected UTM zone %s for %s", crs, inp.name)
            out_path = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            self._warp(inp, out_path, crs, x_res, y_res, resample_alg, src_nodata, dst_nodata)
            outputs.append(FlushedOutput(out_path))

        return StepOutput(outputs=outputs)

    def _warp(
        self,
        inp: Path,
        out_path: Path,
        crs: str,
        x_res: float,
        y_res: float,
        resample_alg: str,
        src_nodata: float,
        dst_nodata: float,
    ) -> None:
        from osgeo import gdal

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds = gdal.Warp(
            str(out_path),
            str(inp),
            dstSRS=crs,
            xRes=x_res,
            yRes=y_res,
            resampleAlg=resample_alg,
            srcNodata=src_nodata,
            dstNodata=dst_nodata,
            tps=True,
            creationOptions=[
                "COMPRESS=DEFLATE",
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "BIGTIFF=IF_SAFER",
            ],
        )
        if ds is None:
            raise RuntimeError(f"gdal.Warp failed for {inp} — check GDAL error log")
        ds = None  # flush and close
