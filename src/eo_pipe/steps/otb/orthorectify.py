"""OrthoRectifyStep — OTB OrthoRectification CLI wrapped as a pipeline step."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import rasterio
import rasterio.crs
import rasterio.warp

from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase


@StepRegistry.register
class OrthoRectifyStep(OTBStepBase):
    """Orthorectify a source image to match the geometry of a reference image.

    Uses the OTB ``OrthoRectification`` application to resample and reproject the
    source so it aligns pixel-for-pixel with the reference.  Supports two output
    grid modes:

    * ``"explicit"`` (default) — rasterio reads the reference to derive ULX/ULY/
      size/spacing which are passed as individual OTB parameters.  Precise
      intersection with the reference bounds; no empty NoData border.
    * ``"orthofit"`` — OTB reads the grid directly from the reference via
      ``outputs.mode=orthofit``.  Simpler when the reference is a pre-built
      spatial template (e.g. rasterised S2 MGRS tile).

    Use :class:`~eo_pipe.pipeline.batch.ParallelBatch` to orthorectify each source
    image independently against the same reference::

        import eo_pipe
        from eo_pipe import PipelineComposition, ParallelBatch
        from pathlib import Path

        ctx = (
            PipelineComposition(workspace=Path("/data/work"))
            .add_step(
                "orthorectify",
                ParallelBatch(),
                ref=Path("reference.tif"),
                interpolator="bco",
                elev_dem=Path("/data/dem/srtm"),
            )
            .run(inputs=[Path("raw_sar_1.tif"), Path("raw_sar_2.tif")])
        )

    Parameters (passed via ``**params`` to :meth:`build_otb_params`):
        ref (Path):
            Reference image.  Its CRS, bounds, and pixel size define the output
            grid. **Required.**
        ref_mode (str):
            Output grid mode — ``"explicit"`` (default) or ``"orthofit"``.
            ``"orthofit"`` delegates all grid computation to OTB.
        interpolator (str):
            OTB resampling interpolator — ``"bco"`` (bicubic, default),
            ``"nn"`` (nearest-neighbour), or ``"linear"``.
        elev_dem (Path | None):
            Directory containing DEM tiles for elevation-aware
            orthorectification.  Skipped when ``None`` (default).
        elev_geoid (Path | None):
            Path to a geoid file (``.bsb``) used for accurate altitude
            correction.  Skipped when ``None`` (default).
        grid_spacing (float):
            Spacing of the OTB resampling grid in pixels.  Smaller values
            improve accuracy at the cost of processing time.  Default: ``4.0``.
        ram_mb (int):
            RAM budget for the OTB application in megabytes.  Default: ``256``.
        compress (bool):
            Write DEFLATE-compressed tiled GeoTIFF via OTB's extended filename
            syntax.  Default: ``True``.  Set to ``False`` to let OTB write its
            default uncompressed output.
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
        """Build ``OrthoRectification`` parameters from the reference image geometry.

        Expected keys in ``params``:
            ref (Path): Reference image whose geometry defines the output grid. Required.
            ref_mode (str): ``"explicit"`` (default) or ``"orthofit"``.
            interpolator (str): OTB resampling method. Default: ``"bco"``.
            elev_dem (Path | None): DEM directory for elevation correction.
            elev_geoid (Path | None): Geoid file for altitude correction.
            grid_spacing (float): OTB resampling-grid spacing in pixels. Default: ``4.0``.
            ram_mb (int): OTB RAM budget in megabytes. Default: ``256``.

        Returns:
            OTB parameter dict (without the output key).
        """
        if len(inputs) != 1:
            raise ValueError(
                f"OrthoRectifyStep expects exactly one input per call; got {len(inputs)}. "
                "Use ParallelBatch to orthorectify multiple files independently."
            )

        ref: Optional[Path] = params.get("ref")
        if ref is None:
            raise TypeError(
                "OrthoRectifyStep requires 'ref' — pass ref=Path(...) to add_step()."
            )

        interpolator: str = params.get("interpolator", "bco")  # type: ignore[assignment]
        elev_dem: Optional[Path] = params.get("elev_dem")  # type: ignore[assignment]
        elev_geoid: Optional[Path] = params.get("elev_geoid")  # type: ignore[assignment]
        grid_spacing: float = params.get("grid_spacing", 4.0)  # type: ignore[assignment]
        ram_mb: int = params.get("ram_mb", 256)  # type: ignore[assignment]
        ref_mode: str = params.get("ref_mode", "explicit")  # type: ignore[assignment]

        if ref_mode not in ("explicit", "orthofit"):
            raise ValueError(
                f"Invalid ref_mode {ref_mode!r}. Choose 'explicit' or 'orthofit'."
            )

        if ref_mode == "orthofit":
            otb_params: Dict[str, Any] = {
                self.param_in: str(inputs[0]),
                "interpolator": interpolator,
                "outputs.mode": "orthofit",
                "outputs.ortho": str(ref),
                "opt.gridspacing": grid_spacing,
                "opt.ram": ram_mb,
            }
        else:
            with rasterio.open(ref) as ref_ds:
                ref_crs = ref_ds.crs
                ref_transform = ref_ds.transform
                ref_bounds = ref_ds.bounds

            with rasterio.open(inputs[0]) as src_ds:
                # OTB-processed files (e.g. SARCalibration output) sometimes
                # lack an embedded CRS. S1 GRD is always WGS84 — fall back.
                src_crs = src_ds.crs or rasterio.crs.CRS.from_epsg(4326)
                src_bounds = rasterio.warp.transform_bounds(
                    src_crs, ref_crs, *src_ds.bounds
                )

            # Intersect source footprint with reference extent
            ix_left = max(ref_bounds.left, src_bounds[0])
            ix_bottom = max(ref_bounds.bottom, src_bounds[1])
            ix_right = min(ref_bounds.right, src_bounds[2])
            ix_top = min(ref_bounds.top, src_bounds[3])

            if ix_left >= ix_right or ix_bottom >= ix_top:
                raise ValueError(
                    f"Source {inputs[0].name!r} has no spatial overlap with"
                    f" reference {ref.name!r}."
                )

            spacing_x = abs(ref_transform.a)
            spacing_y = abs(ref_transform.e)

            # Snap intersection to reference pixel grid
            col0 = math.floor((ix_left - ref_bounds.left) / spacing_x)
            row0 = math.floor((ref_bounds.top - ix_top) / spacing_y)
            col1 = math.ceil((ix_right - ref_bounds.left) / spacing_x)
            row1 = math.ceil((ref_bounds.top - ix_bottom) / spacing_y)

            out_left = ref_bounds.left + col0 * spacing_x
            out_top = ref_bounds.top - row0 * spacing_y

            otb_params = {
                self.param_in: str(inputs[0]),
                "interpolator": interpolator,
                "outputs.ulx": out_left,
                "outputs.uly": out_top,
                "outputs.sizex": col1 - col0,
                "outputs.sizey": row1 - row0,
                "outputs.spacingx": spacing_x,
                "outputs.spacingy": -spacing_y,
                "opt.gridspacing": grid_spacing,
                "opt.ram": ram_mb,
            }

            otb_params.update(_map_projection_params(ref_crs))

        if elev_dem is not None:
            otb_params["elev.dem"] = str(elev_dem)
        if elev_geoid is not None:
            otb_params["elev.geoid"] = str(elev_geoid)

        return otb_params


def _map_projection_params(crs: rasterio.crs.CRS) -> Dict[str, Any]:
    """Translate a rasterio CRS into OTB map-projection parameters.

    Priority order:

    1. Geographic CRS (lat/lon) → ``map=wgs``.
    2. UTM north (EPSG 326xx) or south (EPSG 327xx) → ``map=utm`` with zone
       and hemisphere flag.
    3. Any other projected CRS with a resolvable EPSG code → ``map=epsg``.
    4. Unresolvable CRS → raises :class:`ValueError`; no silent fallback.

    Args:
        crs: Rasterio CRS of the reference image.

    Returns:
        Dict with ``"map"`` key and any required sub-parameters.
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
