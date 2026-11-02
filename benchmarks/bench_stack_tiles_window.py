"""Benchmark stack_tiles: no window vs full window vs thin windows.

Run from the repo root (reuses the test helpers for synthetic data):

    uv run python -m benchmarks.bench_stack_tiles_window --size 4096 --dates 12 --chunk-side 512

With real rasters (one lattice, filename-dated, e.g. Sentinel-1 RTC) pass them with --inputs:

    uv run python -m benchmarks.bench_stack_tiles_window --inputs a_vv.tif b_vv.tif ... --chunk-side 512 --repeats 1

Prints wall time, file size and chunks written per variant. Median of --repeats runs.
"""

import argparse
import math
import shutil
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


def strip_window(nr: int, nc: int, fraction: float) -> np.ndarray:
    """Diagonal band of chunks covering about *fraction* of an nr x nc chunk grid (a thin AOI)."""
    rr, cc = np.mgrid[:nr, :nc]
    dist = np.abs(rr / max(nr - 1, 1) - cc / max(nc - 1, 1)).ravel()
    keep = np.argsort(dist, kind="stable")[: max(1, round(fraction * nr * nc))]
    w = np.zeros(nr * nc, dtype=bool)
    w[keep] = True
    return w.reshape(nr, nc)


def run_once(series: list, out: Path, **params) -> tuple:
    start = time.perf_counter()
    path = StackTilesStep().execute(series, out, **params).flush_all().outputs[0]
    elapsed, size = time.perf_counter() - start, path.stat().st_size
    shutil.rmtree(out)   # real stacks are GBs
    return elapsed, size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048, help="source raster side in pixels")
    ap.add_argument("--dates", type=int, default=8)
    ap.add_argument("--inputs", type=Path, nargs="*", help="real rasters to stack (group mode) instead of synthetic ones")
    ap.add_argument("--inputs-file", type=Path, help="text file of whitespace-separated --inputs paths")
    ap.add_argument("--chunk-side", type=int, default=256)
    ap.add_argument("--memory-budget-mb", type=float, default=4096)
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        inputs = list(args.inputs or [])
        if args.inputs_file:
            inputs += [Path(p) for p in args.inputs_file.read_text().split()]
        if inputs:
            series, extra = inputs, {"group": "bench"}
        else:
            series, extra = build_aligned(Path(tmp), args.size, args.dates), {}
        # The chunk grid comes from the real Cube (inputs are cropped to their intersection).
        probe = StackTilesStep().execute(series, Path(tmp) / "probe", chunk_side=args.chunk_side, **extra)
        ds = probe.outputs[0].ds
        height, width = ds.sizes["y"], ds.sizes["x"]
        nr, nc = (len(c) for c in next(iter(ds.data_vars.values())).chunks[-2:])
        n = nr * nc
        variants = {"no window (baseline)": None, "full window": np.ones((nr, nc), bool)}
        for f in args.fractions:
            variants[f"window {f:.0%}"] = strip_window(nr, nc, f)
        base = dict(chunk_side=args.chunk_side, memory_budget_mb=args.memory_budget_mb, **extra)
        print(f"{len(series)} dates, {width}x{height} px, chunk_side={args.chunk_side} ({nr}x{nc} = {n} chunks)")
        print(f"{'variant':<24}{'chunks':>8}{'time s':>10}{'size MB':>10}{'speedup':>9}")
        baseline = None
        for label, window in variants.items():
            params = dict(base, **({} if window is None else {"window": window}))
            runs = [run_once(series, Path(tmp) / f"out_{label.split()[0]}_{i}", **params) for i in range(args.repeats)]
            t = statistics.median(r[0] for r in runs)
            baseline = baseline or t
            chunks = n if window is None else int(window.sum())
            print(f"{label:<24}{chunks:>8}{t:>10.2f}{runs[0][1] / 2**20:>10.1f}{baseline / t:>8.1f}x")


if __name__ == "__main__":
    main()
