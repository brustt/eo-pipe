"""Benchmark stack_tiles: no window vs full window vs thin windows.

Run from the repo root (reuses the test helpers for synthetic data):

    uv run python -m benchmarks.bench_stack_tiles_window --size 4096 --dates 12 --chunk-side 512

Prints wall time, file size and chunks written per variant. Median of --repeats runs.
"""

import argparse
import math
import statistics
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from eo_pipe.steps.raster.align_tile import AlignToTileStep
from eo_pipe.steps.raster.stack_tiles import StackTilesStep
from tests.conftest import write_tile_grid
from tests.test_steps.test_align_to_tile import make_raster


def build_aligned(work: Path, size: int, dates: int) -> list:
    grid = write_tile_grid(work / "grid" / "tiles.parquet")
    src, aligned = work / "src", work / "aligned"
    src.mkdir()
    aligned.mkdir()
    rng = np.random.default_rng(0)
    srcs = [
        make_raster(
            src / f"s_{date(2025, 1, 1) + timedelta(days=d):%Y%m%d}T051233_vv.tif", 310000.0, 4890000.0,
            size=size, fill=rng.uniform(1, 100, (1, size, size)).astype("float32"),
        )
        for d in range(dates)
    ]
    return AlignToTileStep().execute(srcs, aligned, grid_path=grid).flush_all().outputs


def strip_window(n: int, fraction: float) -> np.ndarray:
    """Diagonal band of chunks covering about *fraction* of an n x n chunk grid (a thin AOI)."""
    w = np.zeros((n, n), dtype=bool)
    target = max(1, round(fraction * n * n))
    for offset in range(n):
        for r in range(n):
            for c in (r + offset, r - offset):
                if 0 <= c < n and w.sum() < target:
                    w[r, c] = True
        if w.sum() >= target:
            break
    return w


def run_once(series: list, out: Path, **params) -> tuple:
    start = time.perf_counter()
    path = StackTilesStep().execute(series, out, **params).flush_all().outputs[0]
    return time.perf_counter() - start, path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048, help="source raster side in pixels")
    ap.add_argument("--dates", type=int, default=8)
    ap.add_argument("--chunk-side", type=int, default=256)
    ap.add_argument("--memory-budget-mb", type=float, default=4096)
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    n = math.ceil(args.size / args.chunk_side)
    variants = {"no window (baseline)": None, "full window": np.ones((n, n), bool)}
    for f in args.fractions:
        variants[f"window {f:.0%}"] = strip_window(n, f)

    with tempfile.TemporaryDirectory() as tmp:
        series = build_aligned(Path(tmp), args.size, args.dates)
        base = dict(chunk_side=args.chunk_side, memory_budget_mb=args.memory_budget_mb)
        print(f"{args.dates} dates, {args.size}x{args.size} px, chunk_side={args.chunk_side} ({n}x{n} = {n * n} chunks)")
        print(f"{'variant':<24}{'chunks':>8}{'time s':>10}{'size MB':>10}{'speedup':>9}")
        baseline = None
        for label, window in variants.items():
            params = dict(base, **({} if window is None else {"window": window}))
            runs = [run_once(series, Path(tmp) / f"out_{label.split()[0]}_{i}", **params) for i in range(args.repeats)]
            t = statistics.median(r[0] for r in runs)
            baseline = baseline or t
            chunks = n * n if window is None else int(window.sum())
            print(f"{label:<24}{chunks:>8}{t:>10.2f}{runs[0][1] / 2**20:>10.1f}{baseline / t:>8.1f}x")


if __name__ == "__main__":
    main()
