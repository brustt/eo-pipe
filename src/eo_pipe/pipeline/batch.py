from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from .base import StepBase, StepResult


class BatchStrategy(ABC):
    """Defines how a step is applied to a collection of inputs."""

    @abstractmethod
    def apply(
        self,
        step: StepBase,
        inputs: List[Path],
        output_dir: Path,
        **params,
    ) -> StepResult:
        """Apply *step* to *inputs* according to this strategy.

        Args:
            step: The step to execute.
            inputs: Current input paths.
            output_dir: Directory for step outputs.
            **params: Step parameters.
@src/eo_pipe/io/raster_io.py#27-47 write more advanced saving method, allowing compression, bigtif, cog tec. This could be an OOP class. Then, it should not be call inside of processing methods. Should it be called to step class implementation or just defined in @src/eo_pipe/pipeline/base.py#27 ? Analyze the way it could be run trhough  batch strategy
        Returns:
            Combined :class:`StepResult` for all processed inputs.
        """
        ...


class ParallelBatch(BatchStrategy):
    """Run the step once per input; collect all outputs into one result.

    This is the most common mode: each input produces exactly one output.
    Execution is sequential; for true concurrency inject a ``ThreadedBatch``
    without modifying any library code.
    """

    def apply(
        self,
        step: StepBase,
        inputs: List[Path],
        output_dir: Path,
        **params,
    ) -> StepResult:
        combined = StepResult()
        for inp in inputs:
            result = step.run([inp], output_dir, **params)
            combined.outputs.extend(result.outputs)
            combined.artifacts.update(result.artifacts)
            combined.metadata.update(result.metadata)
        return combined


class MergeBatch(BatchStrategy):
    """Pass all inputs together as a single call to the step.

    Used for merging steps that need all files at once (e.g. raster merge).
    The step's ``execute`` receives the full ``inputs`` list.
    """

    def apply(
        self,
        step: StepBase,
        inputs: List[Path],
        output_dir: Path,
        **params,
    ) -> StepResult:
        return step.run(inputs, output_dir, **params)


class SingleBatch(BatchStrategy):
    """Assert exactly one input and pass it as a single-element list.

    Raises:
        ValueError: If ``inputs`` does not contain exactly one element.
    """

    def apply(
        self,
        step: StepBase,
        inputs: List[Path],
        output_dir: Path,
        **params,
    ) -> StepResult:
        if len(inputs) != 1:
            raise ValueError(
                f"SingleBatch requires exactly one input, got {len(inputs)}"
            )
        return step.run(inputs, output_dir, **params)
