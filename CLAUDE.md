# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`eo-pipe` is an agnostic Earth Observation (EO) processing pipeline library. It provides composable raster/vector processing steps with a fluent builder API.

## Development Setup

Uses `uv` for package management (Python 3.12).

```bash
uv sync                    # Install dependencies
uv run python <script.py>  # Run a script using the library
uv run pytest              # Run the test suite
```

The package is installed in editable mode via `uv_build` with a `src/` layout. Import as `import eo_pipe` (no `src.` prefix needed after install).

Log level is controlled by the `ENVIRONMENT` env var: `development`=DEBUG (default), `production`=INFO, `testing`=WARNING.

### GDAL rebuild requirement

`osgeo` must be compiled against the system GDAL (installed at `/usr/local/lib`). After creating or recreating the venv, run:

```bash
uv pip install setuptools numpy
LDFLAGS="-L/usr/local/lib -Wl,-rpath,/usr/local/lib" \
  uv pip install --no-build-isolation --reinstall "gdal==3.12.2"
```

See `docs/troubleshooting.md` for the full explanation.

## Architecture

### Core Abstractions

**`StepBase`** ([pipeline/base.py](src/eo_pipe/pipeline/base.py)) — Abstract base for all steps. Subclasses set a `name` class variable and implement `execute(inputs, output_dir, **params) -> StepResult`. The public `run()` wrapper adds timing and logging.

**`StepResult`** — Always returned by every step: `outputs` (List[Path], become next step's inputs), `artifacts` (Dict[str, Path], secondary files), `metadata` (Dict[str, Any], computed values).

**`StepRegistry`** ([pipeline/registry.py](src/eo_pipe/pipeline/registry.py)) — Maps string names to step classes. Steps self-register via `@StepRegistry.register` decorator. Custom steps register the same way without modifying library code.

**`BatchStrategy`** ([pipeline/batch.py](src/eo_pipe/pipeline/batch.py)) — Controls how a step is applied to multiple inputs:
- `ParallelBatch` — runs the step once per input file (default)
- `MergeBatch` — passes all inputs together in a single call
- `SingleBatch` — asserts exactly one input, then passes it

**`PipelineContext`** ([pipeline/context.py](src/eo_pipe/pipeline/context.py)) — Mutable state threaded through a run: current `inputs`, completed step `outputs`, `workspace`, user `metadata`.

**`PipelineComposition`** ([pipeline/composition.py](src/eo_pipe/pipeline/composition.py)) — Fluent builder that chains steps:

```python
import eo_pipe  # triggers step self-registration
from eo_pipe import PipelineComposition, ParallelBatch, MergeBatch
from pathlib import Path

ctx = (
    PipelineComposition(workspace=Path("/data/interim/run_01"))
    .add_step("resample", ParallelBatch(), target_resolution=0.3, method="average")
    .add_step("clip", ParallelBatch(), shp=Path("forest.shp"))
    .add_step("merge_raster", MergeBatch())
    .run(inputs=[Path("a.tif"), Path("b.tif")], save_intermediate=True)
)
final_files = ctx.inputs           # List[Path] — last step's outputs
clip_result = ctx.outputs["clip"]  # StepResult for a specific step
```

When `save_intermediate=False` (default), a temp dir is used and final outputs are copied to `output_dir` (or cwd) before cleanup.

### Registered Steps

Raster steps (`steps/raster/`):

| Name | Class | Key params |
|---|---|---|
| `resample` | `ResampleStep` | `target_resolution`, `method`, `nodata_value` |
| `clip` | `ClipStep` | `shp` (Path or GeoDataFrame), `save_overlay` |
| `merge_raster` | `MergeRasterStep` | `method`, `output_name`, `to_cog`, `nodata` |
| `filter` | `FilterStep` | `method` (`"median"`), `kernel_size` |
| `sieve` | `SieveStep` | `threshold`, `connectedness` — requires system GDAL |
| `calibrate` | `HistogramCalibrationStep` | `ref_path`, `match_proportion`, `color_space` |

Vector steps (`steps/vector/`):

| Name | Class | Key params |
|---|---|---|
| `merge_vector` | `MergeVectorStep` | `output_name`, `driver`, `dissolve` |
| `rasterize` | `RasterizeStep` | `vector_path`, `value_column`, `fill`, `all_touched` |

### Adding a Custom Step

```python
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry
from pathlib import Path
from typing import List

@StepRegistry.register
class MyStep(StepBase):
    name = "my_step"

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        # process inputs, write to output_dir
        return StepResult(outputs=[...])
```

### IO Utilities

- **[io/raster_io.py](src/eo_pipe/io/raster_io.py)** — `RasterWriter` (dataclass, configurable GeoTIFF writer), `DEFAULT_WRITER` (default instance), `downsample_raster`, `hist_match_worker`, `create_hillshade` (requires system GDAL)
- **[io/vector_io.py](src/eo_pipe/io/vector_io.py)** — `save_vector`
- **[io/path_utils.py](src/eo_pipe/io/path_utils.py)** — `PrefixedPathStrategy` (`{step_name}_{input_name}`), `IndexedPathStrategy` (`{step_name}_{idx:03d}_{input_name}`)

`RasterWriter` is the standard way to write GeoTIFFs. Steps use `DEFAULT_WRITER` (deflate, tiled 512×512) or accept a custom `writer=` instance.

### Logging

```python
from eo_pipe.logging import setup_logger, log_step_start, log_step_complete, log_error
logger = setup_logger(__name__)
```

Rich-formatted output; optionally write to rotating log file via `log_file=Path(...)`.

### Testing

Tests use real synthetic 64×64 GeoTIFFs written to `tmp_path` — no rasterio mocking. Run with `uv run pytest`. Tests requiring system GDAL (`sieve`) auto-skip when `osgeo` is unavailable.
