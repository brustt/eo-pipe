from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

from .base import StepBase, StepOutput, StepResult


class BatchStrategy(ABC):
    """Defines how a step is applied to a collection of inputs.

    :meth:`apply` receives a :class:`StepBase` and a list of input paths,
    orchestrates calls to :meth:`~StepBase.run`, flushes all
    :class:`~eo_pipe.io.output_types.Persistable` outputs, and returns a
    :class:`StepResult` with resolved paths.
    """

    @abstractmethod
    def apply(
        self,
        step: StepBase,
        inputs: List[Path],
        output_dir: Path,
        **params: Any,
    ) -> StepResult:
        """Apply *step* to *inputs* according to this strategy.

        Args:
            step: The step to execute.
            inputs: Current input paths.
            output_dir: Directory for step outputs.
            **params: Step parameters.

        Returns:
            :class:`StepResult` with all outputs flushed to disk.
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
        **params: Any,
    ) -> StepResult:
        combined = StepResult()
        for inp in inputs:
            step_out: StepOutput = step.run([inp], output_dir, **params)
            combined.outputs.extend(p.flush() for p in step_out.outputs)
            combined.artifacts.update(step_out.artifacts)
            combined.metadata.update(step_out.metadata)
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
        **params: Any,
    ) -> StepResult:
        return step.run(inputs, output_dir, **params).flush_all()


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
        **params: Any,
    ) -> StepResult:
        if len(inputs) != 1:
            raise ValueError(
                f"SingleBatch requires exactly one input, got {len(inputs)}"
            )
        return step.run(inputs, output_dir, **params).flush_all()
