"""Out-of-core speckle filtering kernels and plumbing shared by ``lee_filter`` and ``quegan_filter``.

The kernels work on NumPy blocks shaped ``(..., rows, cols)`` (leading axes are independent
dates or bands) and use normalised convolution: invalid pixels are left out of every local
statistic.  :func:`filter_lazily` lifts a kernel onto a dask array with a halo of
``window // 2`` so the result does not depend on the chunking.  Linear power is assumed
(ADR-0004); :func:`require_linear` refuses dB inputs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Tuple

import dask.array as da
import numpy as np
from scipy.ndimage import uniform_filter

from eo_pipe.io.memory import ChunkPlan, plan_chunks

#: Equivalent number of looks of Sentinel-1 IW GRDH.
DEFAULT_ENL = 4.4
DEFAULT_WINDOW = 7
#: Bytes per pixel and date of a chunk's float64 working copies (~10 temporaries of the kernel, at OVERHEAD x on top).
_WORK_BYTES = 32


def check_window(window_size: int) -> int:
    window_size = int(window_size)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and at least 3, got {window_size}.")
    return window_size


def require_linear(attrs: Mapping[str, Any], where: str) -> None:
    """Raise if *attrs* declare a dB scale; the speckle filters never convert silently."""
    units = str(attrs.get("units", "")).strip().lower()
    if units in {"db", "decibel", "decibels"}:
        raise ValueError(
            f"{where}: units are dB, but speckle filters need linear power; convert it first "
            "(no silent conversion is done)."
        )


def invalid_mask(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    bad = ~np.isfinite(data)
    if nodata is not None and not math.isnan(nodata):
        bad |= data == nodata
    return bad


def _local_mean(values: np.ndarray, valid: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean over the valid pixels of each window, and the count of them (zero outside the image)."""
    kw = dict(size=window, axes=(-2, -1), mode="constant", cval=0.0)
    n = uniform_filter(valid.astype(np.float64), **kw) * window**2
    total = uniform_filter(values, **kw) * window**2
    mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0.5)
    return mean, n


def lee_block(block: np.ndarray, *, window: int, enl: float, nodata: Optional[float]) -> np.ndarray:
    """Classic Lee filter of a ``(..., rows, cols)`` block; invalid pixels come back as NaN."""
    bad = invalid_mask(block, nodata)
    valid = ~bad
    x = np.where(valid, block, 0.0).astype(np.float64)
    mean, n = _local_mean(x, valid, window)
    mean_sq, _ = _local_mean(x * x, valid, window)
    var = np.maximum(mean_sq - mean**2, 0.0)
    ci2 = np.divide(var, mean**2, out=np.zeros_like(var), where=mean != 0)
    cu2 = 1.0 / enl
    weight = np.divide(ci2 - cu2, ci2, out=np.zeros_like(ci2), where=ci2 > cu2)
    out = mean + weight * (x - mean)
    out[bad] = np.nan
    return out.astype(block.dtype if block.dtype.kind == "f" else np.float32)


def quegan_block(block: np.ndarray, *, window: int, nodata: Optional[float]) -> np.ndarray:
    """Quegan & Yu filter of a ``(time, rows, cols)`` block; invalid pixels come back as NaN."""
    bad = invalid_mask(block, nodata)
    valid = ~bad
    x = np.where(valid, block, 0.0).astype(np.float64)
    mean, _ = _local_mean(x, valid, window)
    usable = valid & (mean > 0)
    ratio = np.divide(x, mean, out=np.zeros_like(x), where=usable)
    count = usable.sum(axis=0)
    mean_ratio = np.divide(ratio.sum(axis=0), count, out=np.ones(count.shape), where=count > 0)
    out = mean * mean_ratio
    out[bad] = np.nan
    return out.astype(block.dtype if block.dtype.kind == "f" else np.float32)


def plan_filter_chunks(budget: float, dates: int, height: int, width: int, window: int) -> ChunkPlan:
    """Chunk side (halo excluded) and concurrency for a filter over *dates* per pixel."""
    halo = window // 2
    plan = plan_chunks(budget, dates * _WORK_BYTES, height, width, halo=halo)
    return ChunkPlan(side=max(plan.side, halo), workers=plan.workers)


def filter_lazily(array: da.Array, kernel: Callable[..., np.ndarray], halo: int, **kwargs: Any) -> da.Array:
    """Apply *kernel* to a ``(..., rows, cols)`` dask array, chunked spatially, with a halo.

    The image edge is not padded: the kernel's own zero-padded normalised convolution
    handles it, identically whatever the chunk size.
    """
    depth = {array.ndim - 2: halo, array.ndim - 1: halo}
    return array.map_overlap(
        kernel, depth=depth, boundary="none", dtype=array.dtype if array.dtype.kind == "f" else np.float32, **kwargs
    )
