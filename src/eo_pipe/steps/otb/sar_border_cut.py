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


def _detect_s1_borders(path: Path) -> Tuple[int, int, int]:
    """Read first and last image rows to detect S1 IW GRD zero-value margins.

    Returns:
        ``(threshold_x, threshold_y_start, threshold_y_end)`` where all three
        are zero when no margins are detected (step can be skipped).

    ``threshold_x`` is the number of consecutive zeros from the right edge of
    the first row (range margin).  ``threshold_y_start`` / ``threshold_y_end``
    are 1 when the corresponding edge row is entirely zero, 0 otherwise.
    """
    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
        first_row = ds.read(1, window=Window(0, 0, w, 1)).ravel().astype(np.float64)
        last_row = ds.read(1, window=Window(0, h - 1, w, 1)).ravel().astype(np.float64)

    # Range margin: trailing zeros from right end of first row
    nz = np.nonzero(first_row[::-1])[0]
    threshold_x = int(nz[0]) if nz.size else w

    # Azimuth margins: 1 iff the full edge row is zero
    threshold_y_start = int(np.all(first_row == 0))
    threshold_y_end = int(np.all(last_row == 0))

    return threshold_x, threshold_y_start, threshold_y_end


@StepRegistry.register
class SARBorderCutStep(OTBStepBase):
    """Zero out S1 IW GRD sawtooth borders using OTB ``ResetMargin``.

    Two-phase execution:

    1. **Python phase** — reads the first and last image rows to detect
       zero-value margins from burst stitching.  If no margins are found,
       the step is skipped and the input path is forwarded unchanged.
    2. **OTB phase** — calls ``otbcli_ResetMargin`` to zero the detected
       margins in range and azimuth.  Requires OTB ≥ 7.3.

    Thresholds can be supplied explicitly to bypass auto-detection::

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
            Range margin width in pixels, zeroed on both left and right sides.
            Auto-detected from first image row when ``None`` (default).
        threshold_y_start (int | None):
            Azimuth start margin in rows.  Auto-detected when ``None``.
        threshold_y_end (int | None):
            Azimuth end margin in rows.  Auto-detected when ``None``.
        compress (bool):
            Write DEFLATE-compressed tiled GeoTIFF.  Default: ``True``.
        ram_mb (int):
            RAM budget in megabytes.  Default: ``256``.
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
        return super().execute(inputs, output_dir, **merged)

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
        """Build ``ResetMargin`` parameter dict from resolved thresholds.

        ``threshold_x``, ``threshold_y_start``, and ``threshold_y_end`` must
        be present in ``params`` — they are injected by :meth:`execute` before
        this method is called.
        """
        ram_mb: int = int(params.get("ram_mb", 256))

        return {
            self.param_in: str(inputs[0]),
            "threshold.x": int(params["threshold_x"]),
            "threshold.y.start": int(params["threshold_y_start"]),
            "threshold.y.end": int(params["threshold_y_end"]),
            "opt.ram": ram_mb,
        }
