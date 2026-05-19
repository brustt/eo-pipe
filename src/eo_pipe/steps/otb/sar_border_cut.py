"""SARBorderCutStep — detect and zero S1 GRD sawtooth margins via OTB ResetMargin."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.pipeline.base import StepOutput
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase

logger = logging.getLogger(__name__)

# S1 IW GRD margin constants (from S1Tiling / CNES reference implementation)
_RANGE_CUT   = 1000   # pixels zeroed on left + right sides
_AZIMUTH_CUT = 1600   # rows zeroed at top + bottom when azimuth border detected
_NODATA_THR  = _RANGE_CUT * 2  # zero-pixel count threshold for azimuth detection


def _detect_s1_borders(path: Path) -> Tuple[int, int, int]:
    """Detect S1 IW GRD azimuth margins using fixed S1Tiling constants.

    Samples rows 100px inside each edge (avoids pure-zero edge rows) and counts
    zero pixels.  Returns fixed ``_AZIMUTH_CUT`` when too many zeros found.
    ``threshold_x`` is always ``_RANGE_CUT`` (fixed, independent of content).
    """
    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
        north = ds.read(1, window=Window(0, 100, w, 1)).ravel()
        south = ds.read(1, window=Window(0, h - 100, w, 1)).ravel()

    crop_north = int(np.sum(north == 0) > _NODATA_THR)
    crop_south = int(np.sum(south == 0) > _NODATA_THR)

    return _RANGE_CUT, _AZIMUTH_CUT * crop_north, _AZIMUTH_CUT * crop_south


@StepRegistry.register
class SARBorderCutStep(OTBStepBase):
    """Zero out S1 IW GRD sawtooth borders using OTB ``ResetMargin``.

    Two-phase execution:

    1. **Python phase** — samples rows 100px inside top/bottom edges to detect
       azimuth margins.  Range margin uses a fixed 1000-pixel constant.
       If all thresholds are zero the step is skipped.
    2. **OTB phase** — calls ``otbcli_ResetMargin`` with ``mode=threshold``.
       Requires OTB ≥ 7.3.

    Example::

        import eo_pipe
        from eo_pipe import PipelineComposition, ParallelBatch
        from pathlib import Path

        ctx = (
            PipelineComposition(workspace=Path("/data/work"))
            .add_step("sar_calibrate", ParallelBatch(), lut="sigma")
            .add_step("sar_cut_borders", ParallelBatch())
            .run(inputs=[Path("s1a-vv-calibrated.tif")])
        )

    Parameters:
        threshold_x (int | None):
            Range margin in pixels (both sides). Default: ``1000``.
        threshold_y_start (int | None):
            Azimuth start margin in rows. Auto-detected when ``None``.
        threshold_y_end (int | None):
            Azimuth end margin in rows. Auto-detected when ``None``.
        compress (bool):
            Write DEFLATE-compressed tiled GeoTIFF. Default: ``True``.
        ram_mb (int):
            RAM budget in megabytes. Default: ``256``.
    """

    name = "sar_cut_borders"
    otb_app = "ResetMargin"

    _COMPRESS_SUFFIX = (
        "?&gdal:co:COMPRESS=DEFLATE"
        "&gdal:co:TILED=YES"
        "&gdal:co:BLOCKXSIZE=512"
        "&gdal:co:BLOCKYSIZE=512"
    )

    def execute(
        self,
        inputs: List[Path],
        output_dir: Path,
        **params: Any,
    ) -> StepOutput:
        """Detect borders, skip if clean, otherwise apply OTB ResetMargin."""
        if len(inputs) != 1:
            raise ValueError(
                f"SARBorderCutStep expects exactly one input per call; got {len(inputs)}. "
                "Use ParallelBatch to process multiple files independently."
            )

        tx: Optional[int] = params.get("threshold_x")
        ty_start: Optional[int] = params.get("threshold_y_start")
        ty_end: Optional[int] = params.get("threshold_y_end")

        if tx is None:
            tx, ty_start, ty_end = _detect_s1_borders(inputs[0])

        if tx == 0 and ty_start == 0 and ty_end == 0:
            logger.warning("No S1 margins detected — passthrough without calling OTB ResetMargin")
            return StepOutput(outputs=[FlushedOutput(inputs[0])])

        merged = {**params, "threshold_x": tx, "threshold_y_start": ty_start, "threshold_y_end": ty_end}
        result = super().execute(inputs, output_dir, **merged)

        # OTB ResetMargin can drop georeference; copy CRS+transform from input if missing.
        out_path = result.outputs[0].path
        with rasterio.open(out_path) as dst_ds:
            has_crs = dst_ds.crs is not None
        if not has_crs:
            with rasterio.open(inputs[0]) as src_ds:
                src_crs = src_ds.crs
                src_transform = src_ds.transform
            if src_crs is not None:
                with rasterio.open(out_path, "r+") as dst_ds:
                    dst_ds.crs = src_crs
                    dst_ds.transform = src_transform
                logger.debug("Copied CRS from input to OTB ResetMargin output")

        return result

    def _format_otb_output(self, out_path: Path, **params: Any) -> str:
        if params.get("compress", True):
            return str(out_path) + self._COMPRESS_SUFFIX
        return str(out_path)

    def build_otb_params(
        self,
        inputs: List[Path],
        output_path: Path,
        **params: Any,
    ) -> Dict[str, Any]:
        ram_mb: int = int(params.get("ram_mb", 256))

        return {
            self.param_in: str(inputs[0]),
            "mode": "threshold",
            "threshold.x": int(params["threshold_x"]),
            "threshold.y.start": int(params["threshold_y_start"]),
            "threshold.y.end": int(params["threshold_y_end"]),
            "opt.ram": ram_mb,
        }
