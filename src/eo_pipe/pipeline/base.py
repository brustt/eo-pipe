from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from eo_pipe.io.output_types import Persistable
from eo_pipe.io.raster_io import DEFAULT_READER, DEFAULT_WRITER, RasterReader, RasterWriter
from eo_pipe.logging import log_error, log_step_complete, log_step_start, setup_logger


@dataclass
class StepResult:
    """Resolved return value from :class:`BatchStrategy` — all outputs written.

    This is the type stored in :class:`~eo_pipe.pipeline.context.PipelineContext`
    and passed between steps as inputs.

    Attributes:
        outputs: Written output paths — become the next step's inputs.
        artifacts: Secondary outputs keyed by a descriptive name
                   (overlay shapefiles, CSVs, intermediate masks, etc.).
        metadata: Computed values (thresholds, damage ratios, stats)
                  that downstream steps may read without re-parsing a file.
    """

    outputs: List[Path] = field(default_factory=list)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutput:
    """Raw return value from :meth:`StepBase.execute` — outputs not yet written.

    Each item in :attr:`outputs` is a :class:`~eo_pipe.io.output_types.Persistable`
    that knows how to write itself to disk.  The
    :class:`~eo_pipe.pipeline.batch.BatchStrategy` calls
    :meth:`~eo_pipe.io.output_types.Persistable.flush` on each item and collects
    the resolved paths into a :class:`StepResult`.

    Use :meth:`flush_all` when calling :meth:`~StepBase.execute` directly
    (e.g. in tests or custom orchestration) to get a :class:`StepResult` with
    all files on disk.

    Attributes:
        outputs: Pending outputs — not yet written.
        artifacts: Secondary outputs already written during ``execute()``
                   (e.g. overlay shapefiles, sidecar files).
        metadata: Computed values forwarded to the context.
    """

    outputs: Sequence[Persistable] = field(default_factory=list)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def flush_all(self) -> StepResult:
        """Flush every pending output and return a resolved :class:`StepResult`.

        Calls :meth:`~eo_pipe.io.output_types.Persistable.flush` on each item
        in :attr:`outputs`, collecting the resulting paths.  Artifacts and
        metadata are forwarded unchanged.

        Returns:
            :class:`StepResult` with all outputs written to disk.
        """
        return StepResult(
            outputs=[p.flush() for p in self.outputs],
            artifacts=self.artifacts,
            metadata=self.metadata,
        )


class StepBase(ABC):
    """Abstract base for all pipeline steps.

    Subclasses must:
    - Set a unique ``name`` class variable used by :class:`StepRegistry`.
    - Implement :meth:`execute` returning a :class:`StepOutput`.

    The public :meth:`run` wrapper adds timing and logging.
    """

    name: ClassVar[str]

    def __init__(
        self,
        writer: Optional[RasterWriter] = None,
        reader: Optional[RasterReader] = None,
        **kwargs: Any,
    ) -> None:
        self._writer = writer or DEFAULT_WRITER
        self._reader = reader or DEFAULT_READER
        self._logger = setup_logger(f"eo_pipe.steps.{self.__class__.__name__}")

    @abstractmethod
    def execute(
        self,
        inputs: List[Path],
        output_dir: Path,
        **params: Any,
    ) -> StepOutput:
        """Core processing logic.  Must be implemented by subclasses.

        Steps must not write to disk directly.  Instead, return a
        :class:`StepOutput` whose :attr:`~StepOutput.outputs` contain
        :class:`~eo_pipe.io.output_types.Persistable` objects.  IO happens
        when the :class:`~eo_pipe.pipeline.batch.BatchStrategy` calls
        :meth:`~StepOutput.flush_all`, or when the caller invokes it directly.

        Artifacts (sidecar files that must be written during processing, e.g.
        overlay shapefiles) are the exception — they may be written inline and
        referenced by path in :attr:`~StepOutput.artifacts`.

        Args:
            inputs: Input file paths for this step.
            output_dir: Directory where outputs should be written.
            **params: Step-specific parameters passed from the composition.

        Returns:
            A :class:`StepOutput` with pending outputs, artifacts, and metadata.
        """
        ...

    def is_available(self) -> bool:
        """Return True if this step's runtime dependencies are satisfied.

        Override in steps that require external binaries or optional packages
        (e.g. OTB CLI, system GDAL). Default assumes no external dependencies.
        """
        return True

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        **params: Any,
    ) -> StepOutput:
        """Public wrapper: timing, logging, and error propagation.

        Args:
            inputs: Input file paths.
            output_dir: Directory for outputs.
            **params: Step-specific parameters.

        Returns:
            :class:`StepOutput` from :meth:`execute`.

        Raises:
            Exception: Re-raises any exception from :meth:`execute` after logging.
        """
        start = time()
        log_step_start(self._logger, self.name, inputs=inputs, output_dir=output_dir)
        try:
            result = self.execute(inputs, output_dir, **params)
            log_step_complete(self._logger, self.name, duration=time() - start)
            return result
        except Exception as exc:
            log_error(self._logger, f"Error in step '{self.name}'", exc)
            raise
