# Future Improvements

This document collects architectural ideas that go beyond the current implementation scope but are worth pursuing as the library matures.

---

## Open questions & short-term backlog

### Use PathStrategy everywhere

`RasterizeStep` hardcodes `output_dir / f"{output_name}_{inp.stem}.tif"` instead of using a `PathStrategy`. Inconsistent path naming makes output locations harder to predict and override. Low-cost fix: adopt `PrefixedPathStrategy` in `RasterizeStep`.

`MergeRasterStep` now uses `NamedPathStrategy` (added to `io/path_utils.py`) — a fixed-name strategy for steps that produce a single output independent of any specific input.

### Examples in notebooks

Valid, but should wait until the public API (`PipelineComposition`, `add_step`, `run`) is stable. Notebooks go stale quickly if written against a moving interface.

### Performance: threading and GPU

Near-term realistic win: `ThreadedBatch` (already mentioned in `ParallelBatch`'s docstring). The bottleneck in EO pipelines is usually disk I/O, not compute — benchmark before optimising.

GPU acceleration (cupy, RAPIDS) is a significant dependency and only beneficial for large arrays at high resolution. At typical working sizes the overhead dominates. Windowed reading (already used internally by `downsample_raster`) is the more impactful streaming optimisation.

---

## Suggested priority order

1. **PathStrategy everywhere** — `RasterizeStep`, `MergeRasterStep`. *(all earlier prerequisites are resolved)*
2. **Backend-agnostic writers** — `Persistable.flush()` is already the single write callsite; plugging in a `WriterBackend` is now straightforward.
3. **Hooks** — independent of the rest; add when observability is needed.
4. **Streaming `PipelineComposition`** — only if end-to-end latency on single items becomes a requirement; generator `execute()` is a consequence of this, not a prerequisite.

---

### 1- Data I/O Next step: backend-agnostic writers

`Persistable.flush()` is now the single write callsite for all raster outputs. Plugging in a cloud `WriterBackend` (S3, GCS) requires only changing the `RasterWriter` instance injected at composition time — no step code changes. See section 3.

--- 
## 2. Generator-based `execute()` for true streaming

### Motivation

For steps processing hundreds of large rasters, the concern is that all output arrays would be alive simultaneously before any is written. A generator interface would allow flushing each output as it is produced, keeping at most one array in memory regardless of input count.

### Why memory is already bounded — and why generators don't change that

`ParallelBatch.apply()` already flushes inside the loop:

```python
for inp in inputs:
    step_out = step.run([inp], output_dir, **params)
    combined.outputs.extend(p.flush() for p in step_out.outputs)  # flush immediately
```

This means at most one `RasterOutput` array is alive per iteration, regardless of how many inputs there are. The loop structure, not the return type, is what bounds memory.

**A `return_data: bool` parameter does not improve on this.** The problems with such a flag:

- `execute()` return type changes at runtime — `StepOutput.outputs` would contain either `RasterOutput` or `FlushedOutput` depending on a caller-supplied flag. The type contract breaks.
- The threshold (e.g. 8 GB) is unknowable before processing: file size on disk does not equal array size after reprojection, resampling, or type conversion.
- It mixes IO policy (when to write) into step params, which describe processing, not IO strategy. Steps that genuinely cannot buffer (windowed readers, GDAL operations) already encode that decision at the implementation level by returning `FlushedOutput`.

**Once a step returns `FlushedOutput`, generators add nothing for memory.** N paths in a list are negligible. The only scenario generators improve is when steps are called with `MergeBatch` and return a large number of `RasterOutput` items — but that is a `MergeBatch` design issue (merge steps receive all inputs at once by definition, and their output is typically a single merged raster).

### What generators actually enable: cross-step streaming

Generators become valuable for a *true streaming pipeline* — one where item N from step 1 is fed directly into step 2 before item N+1 starts:

```python
# Hypothetical StreamingComposition
for inp in inputs:
    for step, batch, params in self._steps:
        inp = step.execute([inp], output_dir, **params)  # next step sees output immediately
```

This is a fundamentally different execution model from the current batch model. It requires a redesigned `PipelineComposition`, not just a change to the step interface. Implement only if end-to-end latency on single items becomes a demonstrated requirement (e.g. real-time ingestion pipelines).

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

### Critique

Confirmed in principle. The earlier prerequisite — a single write callsite — is now satisfied: all raster writes go through `RasterOutput.flush()` → `RasterWriter.write()`. Introducing a `WriterBackend` field on `RasterWriter` is the only change needed; no step or `BatchStrategy` code would change. The remaining open question is API design for streaming writes (rasterio windows vs. full-array), which matters for large COG outputs.

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

### Critique

Confirmed for observability. Two additions the design should address:

- **Immutability**: hooks must receive a read-only view of `StepResult`, not a mutable reference. A hook that accidentally mutates `result.outputs` would silently corrupt the pipeline's input chain for subsequent steps. Pass a copy or a frozen view.
- **Scope boundary**: hooks must never own IO. A hook that triggers a write after `execute()` returns would require all output arrays to survive until the hook fires, recreating the memory problem described in section 1. If the goal is "save after each step", use `Persistable` — not a hook.



### Logging
logger is recreated at each step ?