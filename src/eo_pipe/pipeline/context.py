from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .base import StepResult


@dataclass
class PipelineContext:
    """Mutable state threaded through a :class:`PipelineComposition` run.

    Attributes:
        inputs: Current working inputs; updated after each step from
                ``StepResult.outputs`` (when non-empty).
        outputs: Results of completed steps keyed by step name.
                 If the same step name appears more than once a numeric
                 suffix (``_1``, ``_2``, …) is appended automatically.
        workspace: Base directory used for intermediate file storage.
        metadata: User-supplied metadata; the library never writes to this.
                  Domain wrappers may put ``zone_name``, ``res``, etc. here.
        save_intermediate: Whether intermediate outputs are persisted.
    """

    inputs: List[Path]
    workspace: Path
    outputs: Dict[str, StepResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    save_intermediate: bool = False

    def record_step(self, step_name: str, result: StepResult) -> None:
        """Register a step result, avoiding key collisions."""
        if step_name not in self.outputs:
            self.outputs[step_name] = result
        else:
            idx = 1
            while f"{step_name}_{idx}" in self.outputs:
                idx += 1
            self.outputs[f"{step_name}_{idx}"] = result

    def advance_inputs(self, result: StepResult) -> None:
        """Update ``inputs`` from *result.outputs* if the step produced any."""
        if result.outputs:
            self.inputs = result.outputs
