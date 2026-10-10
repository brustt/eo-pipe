"""OrthoRectifyStep — OTB OrthoRectification CLI wrapped as a pipeline step."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import rasterio
import rasterio.crs
import rasterio.warp
from rasterio.coords import BoundingBox

from eo_pipe.io.lattice import Lattice, intersect
from eo_pipe.io.tiles import CxPolygon, utm_epsg, utm_zone
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase

import logging
logger = logging.getLogger(__name__)


@StepRegistry.register
class OrthoRectifyStep(OTBStepBase):
    """Orthorectify a source image to match the geometry of a reference image.

    Uses the OTB ``OrthoRectification`` application to resample and reproject
    the source so it aligns with the reference.  Two output grid modes
    (``ref_mode``), both of which put independent runs on the same pixel grid
    when they share the same reference:

    * ``"auto"`` (default) — the grid is computed here with rasterio and passed
      to OTB as ``outputs.ulx/uly/sizex/sizey/spacingx/spacingy`` plus the
      ``map*`` projection parameters derived from the reference CRS.

      - Works with ``ref`` or ``ref_extent``.
      - The reference bounds are intersected with the source footprint (GCPs
        for raw S1 GRD), then snapped outward to the reference pixel lattice.
        Outputs are therefore whole-pixel windows of one common grid, but their
        size and origin vary with the source footprint.
      - Raises ``ValueError`` when source and reference do not overlap.

    * ``"orthofit"`` — OTB reads the grid directly from the reference via
      ``outputs.mode=orthofit`` and ``outputs.ortho=<ref>``.

      - Needs ``ref``; ``ref_extent`` is not supported.
      - Output has exactly the reference's extent, size, spacing and CRS,
        so repeated runs are stackable without cropping. Areas not covered by
        the source are filled with ``outputs.default`` (OTB default ``0``); no
        nodata metadata is set.
      - No overlap check is done here; OTB handles it.

    Pixel alignment is not geometric co-registration: pass ``elev_dem`` to
    limit terrain-induced offsets between acquisitions.

    Parameters (passed via ``**params`` to :meth:`build_otb_params`):
        ref (Path | None): Reference image. Required unless ``ref_extent`` is given.
        ref_extent (Polygon | None): Output extent as ``(UL, UR, LR, LL)`` corners in
            WGS 84 ``(lon, lat)``. Used to derive the output grid when ``ref`` is
            not given. Ignored when ``ref`` is given.
        ref_epsg (int | None): Output EPSG when deriving the grid from ``ref_extent``.
            Default: UTM zone of the extent centre.
        ref_resolution (float): Output pixel size (CRS units) when deriving the grid
            from ``ref_extent``. Default: ``10.0``.
        ref_mode (str): ``"auto"`` (default) or ``"orthofit"`` (needs ``ref``).
        interpolator (str): OTB resampling method. Default: ``"bco"``.
        elev_dem (Path | None): DEM directory. Optional.
        elev_geoid (Path | None): Geoid file. Optional.
        grid_spacing (float): OTB ``opt.gridspacing``, spacing of the resampling
            deformation grid in *output map units* (metres, or degrees for a
            geographic CRS), not pixels. OTB suggests about half the DEM
            ground sampling distance. Default: ``4.0``.
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
        src = self._single_input(inputs)
        ref: Optional[Path] = params.get("ref")
        ref_extent: Optional[CxPolygon] = params.get("ref_extent")
        ref_mode: str = params.get("ref_mode", "auto")
        self._validate_reference(ref, ref_extent, ref_mode)

        otb_params: Dict[str, Any] = {
            self.param_in: str(src),
            **_resampling_params(
                params.get("interpolator", "bco"),
                params.get("grid_spacing", 4.0),
                params.get("ram_mb", 256),
            ),
            **_elevation_params(params.get("elev_dem"), params.get("elev_geoid")),
        }

        if ref_mode == "orthofit":
            otb_params.update(_orthofit_params(ref))
        else:
            ref_crs, ref_bounds, res_x, res_y = self._reference_grid(ref, ref_extent, params)
            otb_params.update(
                self._auto_grid_params(src, ref_crs, ref_bounds, res_x, res_y)
            )
        return otb_params

    def _single_input(self, inputs: List[Path]) -> Path:
        if len(inputs) != 1:
            raise ValueError(
                f"{self.name} expects exactly one input per call; got {len(inputs)}. "
                "Use ParallelBatch to orthorectify multiple files independently."
            )
        return inputs[0]

    def _validate_reference(
        self, ref: Optional[Path], ref_extent: Optional[CxPolygon], ref_mode: str
    ) -> None:
        if ref is None and ref_extent is None:
            raise TypeError(
                f"{self.name} requires either 'ref=Path(...)' (reference image) or "
                "'ref_extent=Polygon' — pass one to add_step()."
            )
        if ref_mode not in ("auto", "orthofit"):
            raise ValueError(
                f"ref_mode must be 'auto' or 'orthofit', got {ref_mode!r}."
            )
        if ref_mode == "orthofit" and ref is None:
            raise ValueError(
                f"{self.name}: ref_mode='orthofit' requires 'ref'; ref_extent is not supported."
            )

    def _reference_grid(
        self, ref: Optional[Path], ref_extent: Optional[CxPolygon], params: Dict[str, Any]
    ) -> "tuple[rasterio.crs.CRS, BoundingBox, float, float]":
        """Return ``(crs, bounds, res_x, res_y)`` of the target grid."""
        if ref is not None:
            return _grid_from_reference(ref)
        res = float(params.get("ref_resolution", 10.0))
        crs, bounds = _grid_from_extent(ref_extent, params.get("ref_epsg"), res)
        return crs, bounds, res, res

    def _auto_grid_params(
        self,
        src: Path,
        ref_crs: rasterio.crs.CRS,
        ref_bounds: BoundingBox,
        res_x: float,
        res_y: float,
    ) -> Dict[str, Any]:
        """OTB ``outputs.*`` + ``map*`` params for the ref/source intersection."""
        src_crs, src_bounds = _source_footprint(src, self.name)
        if src_crs != ref_crs:
            src_bounds = rasterio.warp.transform_bounds(src_crs, ref_crs, *src_bounds)

        overlap = intersect(tuple(ref_bounds), src_bounds)
        if overlap is None:
            raise ValueError(
                f"{self.name}: no spatial overlap between source and reference images."
            )
        # Snap to the reference pixel lattice so independent outputs stay aligned.
        lattice = Lattice(ref_bounds.left, ref_bounds.top, res_x, res_y)
        left, bottom, right, top = lattice.snap_outward(overlap)

        return {
            "outputs.ulx": left,
            "outputs.uly": top,
            "outputs.sizex": round((right - left) / res_x),
            "outputs.sizey": round((top - bottom) / res_y),
            "outputs.spacingx": res_x,
            "outputs.spacingy": -res_y,
            **_map_projection_params(ref_crs),
        }


def _resampling_params(interpolator: str, grid_spacing: float, ram_mb: int) -> Dict[str, Any]:
    """``interpolator`` and ``opt.*`` (resampling grid spacing, RAM) parameters."""
    return {
        "interpolator": interpolator,
        "opt.gridspacing": grid_spacing,
        "opt.ram": ram_mb,
    }


def _elevation_params(dem: Optional[Path], geoid: Optional[Path]) -> Dict[str, Any]:
    """Optional ``elev.dem`` / ``elev.geoid`` parameters."""
    out: Dict[str, Any] = {}
    if dem is not None:
        out["elev.dem"] = str(dem)
    if geoid is not None:
        out["elev.geoid"] = str(geoid)
    return out


def _orthofit_params(ref: Path) -> Dict[str, Any]:
    """Let OTB take the output grid straight from *ref* (``outputs.mode=orthofit``)."""
    return {"outputs.mode": "orthofit", "outputs.ortho": str(ref)}


def _grid_from_reference(
    ref: Path,
) -> "tuple[rasterio.crs.CRS, BoundingBox, float, float]":
    """Read CRS, bounds and pixel size from a reference raster."""
    with rasterio.open(ref) as ds:
        return ds.crs, ds.bounds, abs(ds.transform.a), abs(ds.transform.e)


def _source_footprint(src: Path, step_name: str) -> "tuple[rasterio.crs.CRS, tuple]":
    """Return ``(crs, (left, bottom, right, top))`` of the source image.

    Raw S1 GRD has no CRS (GCP-georeferenced only), so the footprint is taken
    from the GCPs instead of the pixel-space bounds.
    """
    with rasterio.open(src) as ds:
        if ds.crs is not None:
            return ds.crs, tuple(ds.bounds)
        gcps, gcp_crs = ds.gcps
        if not gcps:
            raise ValueError(f"{step_name}: {src} has neither a CRS nor GCPs.")
        xs = [g.x for g in gcps]
        ys = [g.y for g in gcps]
        return gcp_crs or rasterio.crs.CRS.from_epsg(4326), (min(xs), min(ys), max(xs), max(ys))


def _grid_from_extent(
    extent: CxPolygon, epsg: Optional[int], resolution: float
) -> "tuple[rasterio.crs.CRS, BoundingBox]":
    """Project a WGS 84 ``(UL, UR, LR, LL)`` extent into *epsg* and snap to *resolution*."""
    lons = [pt[0] for pt in extent]
    lats = [pt[1] for pt in extent]
    if epsg is None:
        epsg = utm_epsg(sum(lons) / len(lons), sum(lats) / len(lats))
    crs = rasterio.crs.CRS.from_epsg(epsg)
    left, bottom, right, top = rasterio.warp.transform_bounds(
        "EPSG:4326", crs, min(lons), min(lats), max(lons), max(lats)
    )
    left, bottom, right, top = Lattice(0.0, 0.0, resolution).snap_outward(
        (left, bottom, right, top)
    )
    return crs, BoundingBox(left, bottom, right, top)


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
        utm = utm_zone(epsg)
        if utm is not None:
            return {"map": "utm", "map.utm.zone": utm[0], "map.utm.northhem": utm[1]}
        return {"map": "epsg", "map.epsg.code": epsg}

    raise ValueError(
        f"Cannot determine OTB projection from CRS — to_epsg() returned None. "
        f"Use an EPSG-resolvable CRS or override build_otb_params. "
        f"CRS WKT: {crs.to_wkt()[:120]}..."
    )
