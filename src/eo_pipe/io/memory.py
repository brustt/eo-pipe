"""Memory budget -> chunk size, shared by every out-of-core step (ADR-0004).

A step hands over its *Memory budget* and the bytes one pixel costs it; the helpers
return how large a spatial chunk may be and how many chunks may run at once.  Peak
memory is an estimate, not a guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: Default Memory budget, in MB, of every out-of-core step.
DEFAULT_MEMORY_BUDGET_MB = 4096.0
#: Smallest square chunk the planner will produce.
MIN_CHUNK_SIDE = 16
#: Multiple of a chunk's payload held at once (input window + output block + write copy).
OVERHEAD = 3.0
#: Prefer at least this chunk side over running more chunks concurrently.
_PREFERRED_SIDE = 128
_MB = 2**20


@dataclass(frozen=True)
class ChunkPlan:
    """Square spatial chunk of ``side`` pixels, ``workers`` of them processed at once."""

    side: int
    workers: int


def budget_mb(params: dict) -> float:
    """The ``memory_budget_mb`` step parameter, validated."""
    value = float(params.get("memory_budget_mb", DEFAULT_MEMORY_BUDGET_MB))
    if not value > 0:
        raise ValueError(f"memory_budget_mb must be positive, got {value}.")
    return value


def _min_budget_error(budget: float, needed_bytes: float) -> ValueError:
    return ValueError(
        f"memory_budget_mb={budget:g} is too small for even one minimal chunk "
        f"({MIN_CHUNK_SIDE} pixel rows or a {MIN_CHUNK_SIDE}x{MIN_CHUNK_SIDE} chunk); set memory_budget_mb to at least "
        f"{needed_bytes / _MB:.3f}."
    )


def plan_chunks(
    budget: float,
    bytes_per_pixel: float,
    height: int,
    width: int,
    max_workers: Optional[int] = None,
    halo: int = 0,
) -> ChunkPlan:
    """Largest square chunk (and concurrency) whose estimated peak fits *budget* MB.

    *bytes_per_pixel* is the payload of one pixel across everything a chunk holds
    (time axis x variables x itemsize).  Concurrency is reduced before the chunk is
    shrunk below a sensible size.

    With a *halo*, each chunk is processed with ``halo`` extra pixels on every side, so
    the returned ``side`` is the chunk *without* them and the budget is checked against
    the padded ``(side + 2 * halo)`` square.

    Raises:
        ValueError: If not even a minimal chunk fits, stating the minimum budget.
    """
    if max_workers is None:
        from dask.system import CPU_COUNT

        max_workers = CPU_COUNT
    per_pixel = bytes_per_pixel * OVERHEAD
    pad = 2 * halo
    minimal = per_pixel * (MIN_CHUNK_SIDE + pad) ** 2
    if budget * _MB < minimal:
        raise _min_budget_error(budget, minimal)
    cap = max(height, width, 1)
    side, workers = MIN_CHUNK_SIDE, 1
    for w in range(max(1, max_workers), 0, -1):
        side = int(math.sqrt(budget * _MB / (w * per_pixel))) - pad
        workers = w
        if side >= min(_PREFERRED_SIDE, cap):
            break
    side = min(side, cap)
    if side > MIN_CHUNK_SIDE:
        side -= side % MIN_CHUNK_SIDE
    return ChunkPlan(side=max(side, 1), workers=workers)


def plan_rows(budget: float, bytes_per_row: float) -> int:
    """Rows (at least ``MIN_CHUNK_SIDE``) of a full-width strip whose estimated peak fits *budget* MB."""
    per_row = bytes_per_row * OVERHEAD
    if budget * _MB < per_row * MIN_CHUNK_SIDE:
        raise _min_budget_error(budget, per_row * MIN_CHUNK_SIDE)
    return int(budget * _MB / per_row)
