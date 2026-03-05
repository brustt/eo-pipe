"""eo-pipe: Agnostic Earth Observation processing pipeline library."""

# Import steps to trigger self-registration with StepRegistry
import eo_pipe.steps  # noqa: F401

from eo_pipe.pipeline import (
    MergeBatch,
    ParallelBatch,
    PipelineComposition,
    PipelineContext,
    SingleBatch,
    StepBase,
    StepRegistry,
    StepResult,
)

__all__ = [
    "PipelineComposition",
    "PipelineContext",
    "StepBase",
    "StepResult",
    "StepRegistry",
    "ParallelBatch",
    "MergeBatch",
    "SingleBatch",
]
