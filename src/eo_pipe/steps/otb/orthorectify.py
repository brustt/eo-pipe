"""OrthoRectifyStep — OTB OrthoRectification CLI wrapped as a pipeline step."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import rasterio
import rasterio.crs
import rasterio.warp

from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase

import logging
logger = logging.getLogger(__name__)


@StepRegistry.register
class OrthoRectifyStep(OTBStepBase):
    """Orthorectify a source image to match the geometry of a reference image.

    Uses the OTB ``OrthoRectification`` application to resample and reproject
    the source so it aligns with the reference.  Two output grid modes:

    * ``"explicit"`` (default) — rasterio reads the reference to derive
      ULX/ULY/size/spacing, clipped to the intersection with the source
      footprint.
    * ``"orthofit"`` — OTB reads the grid directly from the reference via
      ``outputs.mode=orthofit``.

    Parameters (passed via ``**params`` to :meth:`build_otb_params`):
        ref (Path): Reference image. **Required.**
        ref_mode (str): ``"explicit"`` (default) or ``"orthofit"``.
        interpolator (str): OTB resampling method. Default: ``"bco"``.
        elev_dem (Path | None): DEM directory. Optional.
        elev_geoid (Path | None): Geoid file. Optional.
        grid_spacing (float): OTB resampling-grid spacing in pixels. Default: ``4.0``.
        ram_mb (int): OTB RAM budget in megabytes. Default: ``256``.
        compress (bool): Write DEFLATE-compressed tiled GeoTIFF. Default: ``True``.
    """

    name = "orthorectify"
    otb_app = "OrthoRectification"
    param_in = "io.in"
    param_out = "io.out"

    def build_otb_params(
        self,
        inputs: List[Path],
        output_path: Path,
        **params: Any,
    ) -> Dict[str, Any]:
        if len(inputs) != 1:
            raise ValueError(
                f"{self.name} expects exactly one input per call; got {len(inputs)}. "
                "Use ParallelBatch to orthorectify multiple files independently."
            )

        ref: Optional[Path] = params.get("ref")
        if ref is None:
            raise TypeError(
                f"{self.name} requires 'ref=Path(...)' — pass the reference image path to add_step()."
            )

        ref_mode: str = params.get("ref_mode", "explicit")
        if ref_mode not in ("explicit", "orthofit"):
            raise ValueError(
                f"ref_mode must be 'explicit' or 'orthofit', got {ref_mode!r}."
            )

        interpolator: str = params.get("interpolator", "bco")
        elev_dem: Optional[Path] = params.get("elev_dem")
        elev_geoid: Optional[Path] = params.get("elev_geoid")
        grid_spacing: float = params.get("grid_spacing", 4.0)
        ram_mb: int = params.get("ram_mb", 256)

        otb_params: Dict[str, Any] = {
            self.param_in: str(inputs[0]),
            "interpolator": interpolator,
            "opt.gridspacing": grid_spacing,
            "opt.ram": ram_mb,
        }

        if elev_dem is not None:
            otb_params["elev.dem"] = str(elev_dem)
        if elev_geoid is not None:
            otb_params["elev.geoid"] = str(elev_geoid)

        if ref_mode == "orthofit":
            otb_params["outputs.mode"] = "orthofit"
            otb_params["outputs.ortho"] = str(ref)
        else:
            # Derive output grid from ref bounds intersected with source footprint.
            with rasterio.open(ref) as ref_ds:
                ref_crs = ref_ds.crs
                ref_bounds = ref_ds.bounds
                res_x = abs(ref_ds.transform.a)
                res_y = abs(ref_ds.transform.e)

            with rasterio.open(inputs[0]) as src_ds:
                src_crs = src_ds.crs
                src_bounds = src_ds.bounds

            if src_crs != ref_crs:
                src_bounds = rasterio.warp.transform_bounds(src_crs, ref_crs, *src_bounds)

            int_left = max(ref_bounds.left, src_bounds[0])
            int_bottom = max(ref_bounds.bottom, src_bounds[1])
            int_right = min(ref_bounds.right, src_bounds[2])
            int_top = min(ref_bounds.top, src_bounds[3])

            if int_left >= int_right or int_bottom >= int_top:
                raise ValueError(
                    f"{self.name}: no spatial overlap between source and reference images."
                )

            size_x = round((int_right - int_left) / res_x)
            size_y = round((int_top - int_bottom) / res_y)

            otb_params.update({
                "outputs.ulx": int_left,
                "outputs.uly": int_top,
                "outputs.sizex": size_x,
                "outputs.sizey": size_y,
                "outputs.spacingx": res_x,
                "outputs.spacingy": -res_y,
            })
            otb_params.update(_map_projection_params(ref_crs))

        return otb_params


def _map_projection_params(crs: rasterio.crs.CRS) -> Dict[str, Any]:
    """Translate a rasterio CRS into OTB map-projection parameters.

    Priority order:

    1. Geographic CRS (lat/lon) → ``map=wgs``.
    2. UTM north (EPSG 326xx) or south (EPSG 327xx) → ``map=utm`` with zone
       and hemisphere flag.
    3. Any other projected CRS with a resolvable EPSG code → ``map=epsg``.
    4. Unresolvable CRS → raises :class:`ValueError`; no silent fallback.
    """
    if crs.is_geographic:
        return {"map": "wgs"}

    epsg = crs.to_epsg()

    if epsg is not None:
        if 32601 <= epsg <= 32660:
            return {"map": "utm", "map.utm.zone": epsg - 32600, "map.utm.northhem": True}
        if 32701 <= epsg <= 32760:
            return {"map": "utm", "map.utm.zone": epsg - 32700, "map.utm.northhem": False}
        return {"map": "epsg", "map.epsg.code": epsg}

    raise ValueError(
        f"Cannot determine OTB projection from CRS — to_epsg() returned None. "
        f"Use an EPSG-resolvable CRS or override build_otb_params. "
        f"CRS WKT: {crs.to_wkt()[:120]}..."
    )
