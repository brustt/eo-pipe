from pathlib import Path
from typing import Any, List

import numpy as np
from scipy.ndimage import median_filter

from eo_pipe.io.output_types import RasterOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry


@StepRegistry.register
class FilterStep(StepBase):
    """Apply a spatial filter to each input raster.

    Nodata pixels are excluded from the filter computation: after filtering,
    any pixel position that was marked as nodata in the source is restored to
    its original value so the filter does not bleed across nodata boundaries.

    Parameters (passed via ``**params``):
        method (str): Filter algorithm.  Currently only ``"median"``
            is supported.  Defaults to ``"median"``.
        kernel_size (int): Filter kernel size.  Defaults to ``15``.
    """

    name = "filter"

    _METHODS = {"median": "_apply_median"}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        method = params.get("method", "median")
        kernel_size = int(params.get("kernel_size", 15))

        if method not in self._METHODS:
            raise ValueError(
                f"Filter method '{method}' not supported. "
                f"Available: {list(self._METHODS)}"
            )

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            pending = self._build_output(inp, out, method, kernel_size)
            outputs.append(pending)

        return StepOutput(outputs=outputs)

    def _build_output(
        self, inp: Path, out: Path, method: str, kernel_size: int
    ) -> RasterOutput:
        rdata = self._reader.read(inp)
        # TODO: implement a factory for filters methods
        filtered = self._apply_median(rdata.data, kernel_size)

        # Restore original values at nodata positions so the filter kernel
        # does not bleed masked values into valid neighbours.
        if rdata.has_nodata:
            for b in range(filtered.shape[0]):
                filtered[b][~rdata.valid_mask] = rdata.data[b][~rdata.valid_mask]

        return RasterOutput(data=filtered, path=out, meta=rdata.profile, writer=self._writer)

    @staticmethod
    def _apply_median(data: np.ndarray, kernel_size: int) -> np.ndarray:
        if data.ndim == 3:
            result = np.zeros_like(data)
            for b in range(data.shape[0]):
                result[b] = median_filter(data[b], size=kernel_size)
            return result
        return median_filter(data, size=kernel_size)
