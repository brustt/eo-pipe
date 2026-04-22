"""eo-pipe: Agnostic Earth Observation processing pipeline library."""

# Import steps to trigger self-registration with StepRegistry
import eo_pipe.steps  # noqa: F401

from eo_pipe.io.output_types import (
    FlushedOutput,
    GpkgFormat,
    ParquetFormat,
    Persistable,
    RasterOutput,
    ShapefileFormat,
    VectorFormat,
    VectorOutput,
)
from eo_pipe.pipeline import (
    MergeBatch,
    ParallelBatch,
    PipelineComposition,
    PipelineContext,
    SingleBatch,
    StepBase,
    StepOutput,
    StepRegistry,
    StepResult,
)

__all__ = [
    # Pipeline
    "PipelineComposition",
    "PipelineContext",
    "StepBase",
    "StepOutput",
    "StepResult",
    "StepRegistry",
    "ParallelBatch",
    "MergeBatch",
    "SingleBatch",
    # IO output types
    "Persistable",
    "RasterOutput",
    "VectorOutput",
    "FlushedOutput",
    # Vector format strategies
    "VectorFormat",
    "GpkgFormat",
    "ShapefileFormat",
    "ParquetFormat",
]
