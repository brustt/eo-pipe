# eo-pipe

A modular, extensible Python library for building Earth Observation processing pipelines. Compose raster and vector operations into reproducible, chainable workflows — without coupling your code to a specific domain or dataset.

Originally developed through a project at [ONF International](https://www.onf-international.org/) for RGB drone images, it has been completely refactored to make a generic geospatial processing librairy.

## Features

- **Composable steps** — chain raster and vector operations with a fluent builder API
- **Pluggable batch strategies** — process inputs one-by-one or merge them all in a single call
- **Open for extension** — register custom steps with a decorator; no library modification needed
- **Typed return values** — every step returns outputs, side artifacts, and computed metadata
- **COG-ready output** — built-in Cloud-Optimised GeoTIFF support via `RasterWriter`

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd eo-pipe
uv sync
```

> **Note (GDAL):** if your project uses steps that depend on system GDAL (`sieve`), the Python bindings must be compiled against the installed C library. See [docs/troubleshooting.md](docs/troubleshooting.md).

## Quick start

```python
import eo_pipe  # registers all built-in steps
from eo_pipe import PipelineComposition, ParallelBatch, MergeBatch
from pathlib import Path

ctx = (
    PipelineComposition()
    .add_step("resample", ParallelBatch(), target_resolution=0.3, method="average")
    .add_step("clip",     ParallelBatch(), shp=Path("aoi.shp"))
    .add_step("merge_raster", MergeBatch())
    .run(
        inputs=list(Path("data/raw").glob("*.tif")),
        output_dir=Path("data/processed"),
    )
)

print(ctx.inputs)           # final output paths
print(ctx.outputs["clip"])  # StepResult for a specific step
```

## Built-in steps

| Name | Type | Description |
|---|---|---|
| `resample` | raster | Resample to a target resolution |
| `clip` | raster | Clip to a vector geometry |
| `merge_raster` | raster | Merge multiple rasters into one |
| `filter` | raster | Spatial filter (median) |
| `sieve` | raster | Remove small regions (GDAL sieve) |
| `calibrate` | raster | Histogram matching |
| `merge_vector` | vector | Merge vector files |
| `rasterize` | vector | Burn vector values onto a raster grid |
| `orthorectify` | raster | Build orthoimage (OTB) |
| `s1_extract_metadata` | raster (Sentinel 1) | Extract metadata from Sentinel-1 files |
| `s1_calibrate` | raster (Sentinel 1) | Calibrate Sentinel-1 acquisition (OTB) |

## Extending the library

Register a custom step with a single decorator — no changes to library code required:

```python
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry
from pathlib import Path
from typing import List

@StepRegistry.register
class NormalizeStep(StepBase):
    name = "normalize"

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        # your processing logic here
        return StepResult(outputs=[...])
```

Then use it immediately:

```python
PipelineComposition().add_step("normalize", ...)
```

## Development

```bash
uv sync                # install dependencies
uv run pytest          # run the test suite
```

## License

MIT

## Author

Martin Dzr
