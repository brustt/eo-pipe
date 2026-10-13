from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu

from eo_pipe.io.output_types import RasterOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

FilterFn = Callable[[np.ndarray, Optional[np.ndarray], Dict[str, Any]], np.ndarray]


def _median(data: np.ndarray, valid_mask: Optional[np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    kernel_size = int(params.get("kernel_size", 15))
    if data.ndim == 3:
        result = np.zeros_like(data)
        for b in range(data.shape[0]):
            result[b] = median_filter(data[b], size=kernel_size)
        return result
    return median_filter(data, size=kernel_size)


def _otsu(data: np.ndarray, valid_mask: Optional[np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    """Binarise each band (1 above its Otsu threshold, else 0).

    The threshold is computed from valid pixels only.
    """
    bands = data if data.ndim == 3 else data[np.newaxis]
    result = np.zeros_like(bands)
    for b in range(bands.shape[0]):
        band = bands[b]
        values = band[valid_mask] if valid_mask is not None else band.ravel()
        if values.size == 0 or values.min() == values.max():
            continue  # no valid pixels or constant band: nothing to split
        result[b] = band > threshold_otsu(values)
    return result if data.ndim == 3 else result[0]


_FILTER_METHODS: Dict[str, FilterFn] = {"median": _median, "otsu": _otsu}


def get_filter_method(method: str) -> FilterFn:
    """Factory: return the filter callable registered for ``method``."""
    try:
        return _FILTER_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Filter method '{method}' not supported. "
            f"Available: {list(_FILTER_METHODS)}"
        ) from None


@StepRegistry.register
class FilterStep(StepBase):
    """Apply a spatial filter to each input raster.

    Nodata pixels are excluded from the filter computation: after filtering,
    any pixel position that was marked as nodata in the source is restored to
    its original value so the filter does not bleed across nodata boundaries.

    Parameters (passed via ``**params``):
        method (str): Filter algorithm, resolved by ``get_filter_method``:
            ``"median"`` or ``"otsu"`` (binary threshold per band, computed
            on valid pixels).  Defaults to ``"median"``.
        kernel_size (int): Kernel size for ``"median"``.  Defaults to ``15``.
    """

    name = "filter"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        filter_fn = get_filter_method(params.get("method", "median"))

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            pending = self._build_output(inp, out, filter_fn, params)
            outputs.append(pending)

        return StepOutput(outputs=outputs)

    def _build_output(
        self, inp: Path, out: Path, filter_fn: FilterFn, params: Dict[str, Any]
    ) -> RasterOutput:
        rdata = self._reader.read(inp)
        valid_mask = rdata.valid_mask if rdata.has_nodata else None
        filtered = filter_fn(rdata.data, valid_mask, params)

        # Restore original values at nodata positions so the filter kernel
        # does not bleed masked values into valid neighbours.
        if rdata.has_nodata:
            for b in range(filtered.shape[0]):
                filtered[b][~rdata.valid_mask] = rdata.data[b][~rdata.valid_mask]

        return RasterOutput(data=filtered, path=out, meta=rdata.profile, writer=self._writer)
