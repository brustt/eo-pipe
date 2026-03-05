from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Optional

from eo_pipe.io.raster_io import DEFAULT_WRITER, RasterWriter
from eo_pipe.logging import log_error, log_step_complete, log_step_start, setup_logger


@dataclass
class StepResult:
    """Typed return value for every pipeline step.

    Attributes:
        outputs: Primary output paths — become the next step's inputs.
        artifacts: Secondary outputs keyed by a descriptive name
                   (overlay shapefiles, CSVs, intermediate masks, etc.).
        metadata: Computed values (thresholds, damage ratios, stats)
                  that downstream steps may read without re-parsing a file.
    """

    outputs: List[Path] = field(default_factory=list)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StepBase(ABC):
    """Abstract base for all pipeline steps.

    Subclasses must:
    - Set a unique ``name`` class variable used by :class:`StepRegistry`.
    - Implement :meth:`execute` returning a :class:`StepResult`.

    The public :meth:`run` wrapper adds timing and logging.
    """

    name: ClassVar[str]

    def __init__(self, writer: Optional[RasterWriter] = None) -> None:
        self._writer = writer or DEFAULT_WRITER
        self._logger = setup_logger(f"eo_pipe.steps.{self.__class__.__name__}")

    @abstractmethod
    def execute(
        self,
        inputs: List[Path],
        output_dir: Path,
        **params,
    ) -> StepResult:
        """Core processing logic.  Must be implemented by subclasses.

        Args:
            inputs: Input file paths for this step.
            output_dir: Directory where outputs should be written.
            **params: Step-specific parameters passed from the composition.

        Returns:
            A :class:`StepResult` with outputs, artifacts, and metadata.
        """
        ...

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        **params,
    ) -> StepResult:
        """Public wrapper: timing, logging, and error propagation.

        Args:
            inputs: Input file paths.
            output_dir: Directory for outputs.
            **params: Step-specific parameters.

        Returns:
            :class:`StepResult` from :meth:`execute`.

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
