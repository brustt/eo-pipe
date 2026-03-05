from pathlib import Path
from typing import List

import numpy as np
import rasterio as rio
from scipy.ndimage import median_filter

from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry


@StepRegistry.register
class FilterStep(StepBase):
    """Apply a spatial filter to each input raster.

    Parameters (passed via ``**params``):
        method (str): Filter algorithm.  Currently only ``"median"``
            is supported.  Defaults to ``"median"``.
        kernel_size (int): Filter kernel size.  Defaults to ``15``.
    """

    name = "filter"

    _METHODS = {"median": "_apply_median"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
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
            self._filter_raster(inp, out, method, kernel_size)
            outputs.append(out)

        return StepResult(outputs=outputs)

    def _filter_raster(
        self, inp: Path, out: Path, method: str, kernel_size: int
    ) -> None:
        with rio.open(inp) as src:
            data = src.read()
            meta = src.meta.copy()

        filtered = self._apply_median(data, kernel_size)
        self._writer.write(output_file=out, data=filtered, **meta)

    @staticmethod
    def _apply_median(data: np.ndarray, kernel_size: int) -> np.ndarray:
        if data.ndim == 3:
            result = np.zeros_like(data)
            for b in range(data.shape[0]):
                result[b] = median_filter(data[b], size=kernel_size)
            return result
        return median_filter(data, size=kernel_size)
