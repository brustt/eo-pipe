"""Sentinel-1 SAFE product extraction step.

Unzips a Sentinel-1 GRD product, extracts the requested polarisation band(s),
reads ground control points from the annotation XML, and writes a georeferenced
GeoTIFF per band to ``output_dir/{product_name}/{POL}.tif``.
"""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, List

import rasterio
import rasterio.control
import rasterio.crs

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.io.raster_io import DEFAULT_WRITER
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

logger = setup_logger("eo_pipe.steps.raster.s1_extract")

_VALID_POLS: list[str] = ["vv", "vh", "hh", "hv"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_measurement(safe_dir: Path, pol: str) -> Path | None:
    """Return the measurement tiff for *pol* inside *safe_dir*, or None."""
    candidates = list((safe_dir / "measurement").glob(f"*-{pol}-*.tif*"))
    return candidates[0] if candidates else None


def _find_annotation(safe_dir: Path, pol: str) -> Path | None:
    """Return the annotation XML for *pol* (top-level annotation/ only)."""
    candidates = [
        p for p in (safe_dir / "annotation").glob(f"*-{pol}-*.xml")
        if p.parent.name == "annotation"
    ]
    return candidates[0] if candidates else None


def _copy_annotation_tree(safe_dir: Path, pol: str, ann_out_dir: Path) -> None:
    """Copy all annotation files for *pol* — including annotation/calibration/ — to *ann_out_dir*.

    OTB SARCalibration resolves LUTs from annotation/calibration/calibration-*-{pol}-*.xml.
    Copying only the top-level XML leaves those unreachable.
    """
    src_ann_dir = safe_dir / "annotation"
    for src_file in src_ann_dir.rglob(f"*-{pol}-*"):
        rel = src_file.relative_to(src_ann_dir)
        dst = ann_out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)


def _parse_gcps(ann_xml: Path) -> list[rasterio.control.GroundControlPoint]:
    """Extract GCPs from a S1 annotation XML geolocationGrid."""
    tree = ET.parse(ann_xml)
    root = tree.getroot()
    gcps: list[rasterio.control.GroundControlPoint] = []
    for i, pt in enumerate(root.findall(".//geolocationGridPoint")):
        row = float(pt.findtext("line") or 0)
        col = float(pt.findtext("pixel") or 0)
        lat = float(pt.findtext("latitude") or 0)
        lon = float(pt.findtext("longitude") or 0)
        gcps.append(rasterio.control.GroundControlPoint(
            row=row, col=col, x=lon, y=lat, id=str(i)
        ))
    return gcps


def _extract_safe(zip_path: Path, extract_dir: Path, polarisations: list[str]) -> Path:
    """Selectively extract measurement + annotation files for *polarisations*."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # SAFE dir is the first path component (e.g. "S1C_IW_GRDH_...SAFE/")
        safe_name = names[0].split("/")[0]

        for name in names:
            parts = name.split("/")
            # SAFE root files (manifest.safe, etc.) — len==2, non-empty filename
            if len(parts) == 2 and parts[1]:
                zf.extract(name, extract_dir)
                continue
            if len(parts) < 3:
                continue
            subdir = parts[1]
            filename = parts[-1].lower()
            if subdir in ("measurement", "annotation") and any(
                f"-{pol}-" in filename for pol in polarisations
            ):
                zf.extract(name, extract_dir)

        return extract_dir / safe_name


def _write_band(src_tif: Path, out_path: Path, gcps: list) -> Path:
    """Copy pixel data from *src_tif* to *out_path* and embed *gcps*."""
    with rasterio.open(src_tif) as src:
        data = src.read()
        profile = src.meta.copy()

    return DEFAULT_WRITER.write(
        out_path,
        data,
        gcps=gcps or None,
        gcp_crs=rasterio.crs.CRS.from_epsg(4326) if gcps else None,
        **profile,
    )


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


@StepRegistry.register
class S1ExtractStep(StepBase):
    """Extract polarisation bands from a Sentinel-1 SAFE product.

    Accepts either a ``.zip`` archive or an already-extracted ``.SAFE``
    directory.  Writes one GeoTIFF per requested polarisation to::

        output_dir/{product_name}/{POL}.tif

    Ground control points parsed from the annotation XML are embedded in each
    output file so that downstream steps (calibration, orthorectification) can
    geo-reference the raster without the original SAFE package.

    Parameters:
        polarisations (list[str]): Polarisation IDs to extract, e.g.
            ``["VH", "VV"]``. Case-insensitive. Defaults to
            ``["VV", "VH", "HH", "HV"]`` — only those present in the product
            are written.
    """

    name = "s1_extract"

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        raw_pols = params.get("polarisations", params.get("polarisation", _VALID_POLS))
        polarisations = [p.lower() for p in raw_pols]

        unknown = set(polarisations) - set(_VALID_POLS)
        if unknown:
            raise ValueError(
                f"Unknown polarisation(s): {unknown}. Valid: {sorted(_VALID_POLS)}"
            )

        extract_safe: bool = bool(params.get("extract_safe", True))

        outputs: list[FlushedOutput] = []
        for inp in inputs:
            for path in self._process_product(inp, output_dir, polarisations, extract_safe):
                outputs.append(FlushedOutput(path))

        return StepOutput(outputs=outputs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_product(
        self, inp: Path, output_dir: Path, polarisations: list[str], extract_safe: bool
    ) -> list[Path]:
        with tempfile.TemporaryDirectory(prefix="eo_pipe_s1_") as tmp:
            safe_dir = self._resolve_safe(inp, Path(tmp), polarisations)
            product_out = output_dir / safe_dir.stem
            product_out.mkdir(parents=True, exist_ok=True)

            # Preserve SAFE layout so OTB/GDAL can resolve calibration LUTs.
            # manifest.safe at root → identifies product type.
            # annotation/calibration/ → radiometric LUT XMLs.
            for root_file in safe_dir.glob("*"):
                if root_file.is_file():
                    shutil.copy2(root_file, product_out / root_file.name)

            meas_out_dir = product_out / "measurement"
            ann_out_dir = product_out / "annotation"

            written: list[Path] = []
            for pol in polarisations:
                meas = _find_measurement(safe_dir, pol)
                if meas is None:
                    logger.debug(
                        "No %s measurement in %s — skipping", pol.upper(), safe_dir.name
                    )
                    continue

                out_path = meas_out_dir / meas.name

                ann = _find_annotation(safe_dir, pol)
                if ann is not None:
                    _copy_annotation_tree(safe_dir, pol, ann_out_dir)

                if not extract_safe:
                    self._fast_copy(meas, out_path)
                else:
                    if ann is None:
                        logger.warning(
                            "No annotation XML for %s in %s — writing without GCPs",
                            pol.upper(), safe_dir.name,
                        )
                        gcps: list = []
                    else:
                        gcps = _parse_gcps(ann)
                        logger.debug(
                            "Parsed %d GCPs for %s/%s", len(gcps), safe_dir.name, pol.upper()
                        )
                    _write_band(meas, out_path, gcps)

                logger.info("Extracted %s → %s", meas.name, out_path)
                written.append(out_path)

        return written

    def _fast_copy(self, src_tif: Path, out_path: Path) -> None:
        """Copy measurement TIF as-is — no GCP extraction, no re-encoding."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tif, out_path)

    def _resolve_safe(
        self, inp: Path, tmp_dir: Path, polarisations: list[str]
    ) -> Path:
        if inp.suffix.lower() == ".zip":
            return _extract_safe(inp, tmp_dir, polarisations)
        if inp.is_dir() and inp.suffix.upper() == ".SAFE":
            return inp
        raise ValueError(
            f"Input must be a .zip S1 product or a .SAFE directory, got: {inp}"
        )
