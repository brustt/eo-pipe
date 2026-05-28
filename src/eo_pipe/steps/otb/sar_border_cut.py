"""SARBorderCutStep — detect and zero S1 GRD sawtooth margins via OTB ResetMargin."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.pipeline.base import StepOutput
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase

logger = logging.getLogger(__name__)

def _detect_s1_borders(path: Path) -> Tuple[int, int, int]:
    """Detect zero-filled border margins by counting consecutive all-zero lines.

    Returns ``(tx, ty_start, ty_end)`` where:
    - ``tx`` — consecutive all-zero columns from the right (range margin)
    - ``ty_start`` — consecutive all-zero rows from the top (azimuth start)
    - ``ty_end`` — consecutive all-zero rows from the bottom (azimuth end)

    A column or row must be entirely zero to count.
    """
    with rasterio.open(path) as ds:
        data = ds.read(1)

    h, w = data.shape

    tx = 0
    for i in range(w - 1, -1, -1):
        if np.all(data[:, i] == 0):
            tx += 1
        else:
            break

    ty_s = 0
    for i in range(h):
        if np.all(data[i, :] == 0):
            ty_s += 1
        else:
            break

    ty_e = 0
    for i in range(h - 1, -1, -1):
        if np.all(data[i, :] == 0):
            ty_e += 1
        else:
            break

    return tx, ty_s, ty_e


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
    _apply_gdal_options = False

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

    def build_otb_params(
        self,
        inputs: List[Path],
        output_path: Path,
        **params: Any,
    ) -> Dict[str, Any]:
        return {
            self.param_in: str(inputs[0]),
            "mode": "threshold",
            "threshold.x": int(params["threshold_x"]),
            "threshold.y.start": int(params["threshold_y_start"]),
            "threshold.y.end": int(params["threshold_y_end"]),
            "ram": params.get("ram_mb", 256),
        }
