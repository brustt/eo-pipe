# Future Improvements

This document collects architectural ideas that go beyond the current implementation scope but are worth pursuing as the library matures.

---

### Notes
### To do                                                                 
* keep or remove wrapped function ?                                       
* homogenize saving : sometimes in Step class, sometime in the function wrapped by the class                                                        
* Does it could be better to write some callback function, auto run by
    - pipelineComposition ?                                                    
* use PathStrategy everywhere                                             
* make some example in notebooks                                          
* speed tests over methods and propose some gpu optimization/ multithread

## 1. `Persistable` Protocol — separating data from IO

### Motivation

Currently each step is responsible for two distinct concerns: processing data and writing it to disk. The `_writer` injection (introduced in `StepBase`) decouples *which* writer is used, but writing still happens inside `execute()`. A `Persistable` protocol would push IO responsibility fully out of steps, making them pure data transforms that are easier to test, compose, and reason about.

### Design

```python
from typing import Protocol, runtime_checkable
from pathlib import Path

@runtime_checkable
class Persistable(Protocol):
    path: Path  # where the output should be saved

    def flush(self, writer: "RasterWriter") -> Path:
        """Write data to disk and return the saved path."""
        ...
```

Concrete implementations per output type:

```python
@dataclass
class RasterOutput:
    data: np.ndarray
    path: Path
    meta: dict  # rasterio profile (crs, transform, dtype, …)

    def flush(self, writer: RasterWriter) -> Path:
        writer.write(self.path, self.data, **self.meta)
        return self.path

@dataclass
class VectorOutput:
    gdf: gpd.GeoDataFrame
    path: Path

    def flush(self, writer: RasterWriter) -> Path:  # writer is ignored
        self.gdf.to_file(self.path)
        return self.path
```

`StepResult.outputs` would become `List[Persistable]`. `BatchStrategy.apply()` calls `out.flush(writer)` immediately after each file, so only one output lives in memory at a time:

```python
class ParallelBatch(BatchStrategy):
    def apply(self, step, inputs, output_dir, **params):
        combined = StepResult()
        for pending in step.execute(inputs, output_dir, **params):
            path = pending.flush(self._writer)
            combined.outputs.append(path)
        return combined
```

The composition no longer needs to know about data types — it just calls `flush()` on whatever the step returns.

### Why it is not implemented yet

- Requires changing `execute()` to a **generator** (yielding one `Persistable` per file), which is a breaking change to the entire step interface.
- Steps that use rasterio's windowed reading internally (e.g. `downsample_raster`) already stream efficiently; materialising a full array just to wrap it in `RasterOutput` would undo that.
- The current `_writer` injection already covers the primary use case (swapping compression, format, COG output) at low cost.

### When to consider it

- When testing steps requires mocking disk writes frequently.
- When a step needs to produce outputs to different backends (local FS, S3, GCS) within the same run.
- When introducing a `ThreadedBatch` that processes files concurrently — `Persistable.flush()` provides a natural synchronisation point.

---

## 2. Generator-based `execute()` for true streaming

### Motivation

`ParallelBatch` already processes one file at a time by calling `step.run([inp], ...)` in a loop. But `execute()` still collects all outputs into a list before returning. For steps with many outputs, this means all processed arrays are alive until the method returns.

A generator interface would allow flushing each output as it is produced:

```python
class ClipStep(StepBase):
    def execute(self, inputs, output_dir, **params):
        for inp in inputs:
            # … process …
            yield RasterOutput(data=out_image, path=out, meta=out_meta)
```

This pairs naturally with the `Persistable` protocol above — `BatchStrategy` iterates the generator and flushes immediately, keeping at most one array in memory regardless of input count.

### Trade-off

Generator `execute()` is a breaking API change. A transitional approach is to keep the current return type but add an optional `stream_execute()` generator method that `BatchStrategy` prefers when present.

---

## 3. Backend-agnostic writers

### Motivation

`RasterWriter` writes local GeoTIFFs. A `WriterBackend` abstraction would allow plugging in cloud storage (S3 via `s3fs`, GCS via `gcsfs`) without changing any step code.

```python
class WriterBackend(Protocol):
    def open_write(self, path: Path | str, **profile) -> ContextManager:
        """Return a rasterio-compatible write context."""
        ...

@dataclass
class RasterWriter:
    backend: WriterBackend = LocalBackend()
    compress: Compression = "deflate"
    # …
```

This is straightforward once `Persistable.flush()` is the single write callsite — swap the backend on the writer, and all steps pick it up automatically.

---

## 4. Post-step hooks in `PipelineComposition`

### Motivation

Lifecycle callbacks would allow observability (progress bars, metrics collection, artifact inspection) without modifying step code.

```python
class PipelineComposition:
    def add_hook(self, event: str, fn: Callable[[str, StepResult], None]) -> Self:
        """Register a callback for a lifecycle event.

        Events: ``"after_step"``, ``"before_step"``.
        """
        ...
```

Usage:

```python
def log_artifacts(step_name, result):
    for key, path in result.artifacts.items():
        print(f"[{step_name}] artifact saved: {key} → {path}")

pipeline.add_hook("after_step", log_artifacts)
```

### Scope

Hooks are suitable for observability only. They should never own IO responsibility — writing must remain synchronous and inside the step (or inside `flush()` once `Persistable` is adopted). A hook that triggers a write after `execute()` returns would require all output arrays to survive until the hook runs, recreating the memory problem described above.
