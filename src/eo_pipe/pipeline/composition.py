import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import StepBase, StepResult
from .batch import BatchStrategy, ParallelBatch
from .context import PipelineContext
from .registry import StepRegistry
from eo_pipe.io.raster_io import DEFAULT_WRITER, RasterWriter
from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.composition")


class PipelineComposition:
    """Fluent builder that chains pipeline steps into a reproducible run.

    Steps are added with :meth:`add_step` and executed with :meth:`run`.
    Each step receives the previous step's ``StepResult.outputs`` as its
    inputs (or the original inputs for the first step).

    Args:
        workspace: Base directory for intermediate outputs.  When
                   *save_intermediate* is ``False`` in :meth:`run`, a
                   temporary directory is used instead.
        metadata: Arbitrary user metadata attached to :class:`PipelineContext`.
                  The library never reads from this dict; it is only for
                  domain wrappers (e.g. ``zone_name``, ``res``).
        path_strategy: Strategy for generating per-file output paths.
                       Defaults to :class:`PrefixedPathStrategy`.

    Example::

        ctx = (
            PipelineComposition(workspace=Path("/data/interim/run_01"))
            .add_step("resample", ParallelBatch(), target_resolution=0.3)
            .add_step("clip", ParallelBatch(), shp=Path("forest.shp"))
            .add_step("merge_raster", MergeBatch())
            .run(inputs=[Path("a.tif"), Path("b.tif")], save_intermediate=True)
        )
        final = ctx.inputs[0]
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        writer: Optional[RasterWriter] = None,
    ) -> None:
        self._steps: List[Tuple[StepBase, BatchStrategy, Dict[str, Any]]] = []
        self._workspace = workspace
        self._metadata = metadata or {}
        self._writer = writer or DEFAULT_WRITER

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_step(
        self,
        step: Union[str, StepBase],
        batch: Optional[BatchStrategy] = None,
        **params: Any,
    ) -> "PipelineComposition":
        """Append a step to the composition.

        Args:
            step: Either a registered step name (``str``) or a
                  pre-instantiated :class:`StepBase`.
            batch: Batch strategy; defaults to :class:`ParallelBatch`.
            **params: Parameters forwarded to ``step.execute`` at run time.

        Returns:
            ``self`` for chaining.
        """
        if isinstance(step, str):
            step = StepRegistry.create(step, writer=self._writer)
        if batch is None:
            batch = ParallelBatch()
        self._steps.append((step, batch, params))
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        inputs: List[Path],
        output_dir: Optional[Path] = None,
        save_intermediate: bool = False,
        preflight: bool = True,
    ) -> PipelineContext:
        """Execute all steps in order and return the resulting context.

        When *save_intermediate* is ``True``, intermediate outputs are kept
        in ``workspace`` (or a timestamped sub-directory if none was provided).
        When ``False``, a temporary directory is used and cleaned up; the
        final outputs are copied to *output_dir* (or the current working
        directory) before cleanup so they are not lost.

        Args:
            inputs: Initial input paths.
            output_dir: Where to copy final outputs when
                        *save_intermediate* is ``False``.
            save_intermediate: Persist all intermediate step outputs.
            preflight: When ``True`` (default), call ``step.is_available()``
                       for every step before processing any inputs and raise
                       :class:`RuntimeError` listing all unavailable steps.

        Returns:
            :class:`PipelineContext` with ``inputs`` pointing to the final
            outputs and ``outputs`` containing every step's result.
        """
        if preflight:
            unavailable = [
                step.name for step, _, _ in self._steps if not step.is_available()
            ]
            if unavailable:
                raise RuntimeError(
                    "Pipeline preflight failed — the following steps are not available"
                    " in the current environment:\n"
                    + "\n".join(f"  • {name}" for name in unavailable)
                )

        if save_intermediate:
            workspace = self._workspace or Path.cwd() / "eo_pipe_run"
            workspace.mkdir(parents=True, exist_ok=True)
            return self._execute(inputs, workspace, save_intermediate=True)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                ctx = self._execute(inputs, Path(tmp), save_intermediate=False)
                # Copy final outputs out before the temp dir is deleted
                dest = output_dir or Path.cwd()
                dest.mkdir(parents=True, exist_ok=True)
                final_outputs = []
                for p in ctx.inputs:
                    target = dest / p.name
                    shutil.copy2(p, target)
                    final_outputs.append(target)
                ctx.inputs = final_outputs
            return ctx

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(
        self,
        inputs: List[Path],
        workspace: Path,
        save_intermediate: bool,
    ) -> PipelineContext:
        ctx = PipelineContext(
            inputs=list(inputs),
            workspace=workspace,
            metadata=dict(self._metadata),
            save_intermediate=save_intermediate,
        )

        for step_idx, (step, batch, params) in enumerate(self._steps):
            step_dir = workspace / f"{step_idx:02d}_{step.name}"
            step_dir.mkdir(parents=True, exist_ok=True)

            result: StepResult = batch.apply(step, ctx.inputs, step_dir, **params)

            ctx.record_step(step.name, result)
            ctx.advance_inputs(result)

        return ctx
