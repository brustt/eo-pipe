"""Abstract base for OTB-backed pipeline steps."""

from __future__ import annotations

import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepOutput


class OTBStepBase(StepBase):
    """Abstract base for steps that delegate processing to an OTB CLI application.

    Subclasses must declare:

    - ``name`` (``ClassVar[str]``) — unique registry key (inherited from StepBase).
    - ``otb_app`` (``ClassVar[str]``) — OTB application name, e.g.
      ``"OrthoRectification"``.
    - Override ``param_in`` / ``param_out`` when the application uses non-default
      parameter names (OrthoRectification uses ``"io.in"`` / ``"io.out"``).
    - Implement :meth:`build_otb_params` returning the parameter dict.

    The step invokes ``otbcli_{otb_app}`` via :func:`subprocess.run` and wraps
    the written output file in a :class:`~eo_pipe.io.output_types.FlushedOutput`,
    keeping full compatibility with all existing
    :class:`~eo_pipe.pipeline.batch.BatchStrategy` types.

    The CLI binary is resolved at execution time via :func:`shutil.which` — the
    step self-registers normally even when OTB is not installed and raises a clear
    :class:`RuntimeError` at execution time.

    Example::

        @StepRegistry.register
        class MyOTBStep(OTBStepBase):
            name = "my_otb_step"
            otb_app = "BandMath"

            def build_otb_params(self, inputs, output_path, *, expression, **_):
                return {
                    self.param_in: [str(inputs[0])],
                    "exp": expression,
                }
    """

    otb_app: ClassVar[str]
    param_in: ClassVar[str] = "in"
    param_out: ClassVar[str] = "out"

    _COMPRESS_SUFFIX = (
        "?&gdal:co:COMPRESS=DEFLATE"
        "&gdal:co:BIGTIFF=YES"
        "&gdal:co:NUM_THREADS=ALL_CPUS"
        "&gdal:co:TILED=YES"
        "&gdal:co:BLOCKXSIZE=256"
        "&gdal:co:BLOCKYSIZE=256"
    )

    def is_available(self) -> bool:
        return shutil.which(f"otbcli_{self.otb_app}") is not None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(
        self,
        inputs: List[Path],
        output_dir: Path,
        **params: Any,
    ) -> StepOutput:
        """Run the OTB CLI application and return a :class:`FlushedOutput`.

        Resolves ``otbcli_{otb_app}`` on PATH, builds the parameter dict via
        :meth:`build_otb_params`, flattens it into CLI arguments, and invokes
        the application with :func:`subprocess.run`.

        One CLI call is made per ``execute`` invocation. Use
        :class:`~eo_pipe.pipeline.batch.ParallelBatch` to process inputs
        individually; use :class:`~eo_pipe.pipeline.batch.MergeBatch` for
        multi-input OTB applications (e.g. Synthetize).

        Args:
            inputs: Input file paths for this invocation.
            output_dir: Directory where the output file will be written.
            **params: Step-specific parameters forwarded to :meth:`build_otb_params`.

        Returns:
            :class:`~eo_pipe.pipeline.base.StepOutput` wrapping a single
            :class:`~eo_pipe.io.output_types.FlushedOutput`.

        Raises:
            TypeError: If ``otb_app`` class variable is not set.
            RuntimeError: If ``otbcli_{otb_app}`` is not found on PATH or exits
                non-zero.
        """
        if not getattr(self, "otb_app", None):
            raise TypeError(
                f"{self.__class__.__name__} must define an 'otb_app' class variable."
            )

        cmd_name = f"otbcli_{self.otb_app}"
        if shutil.which(cmd_name) is None:
            raise RuntimeError(
                f"OTB CLI command '{cmd_name}' not found on PATH. Install OrfeoToolbox"
                " and ensure its bin/ directory is on PATH."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._path_strategy.resolve(self.name, inputs[0], 0, output_dir)

        otb_params = self.build_otb_params(inputs, out_path, **params)
        if self.param_out in otb_params:
            raise ValueError(
                f"{self.__class__.__name__}.build_otb_params must not set the output key "
                f"'{self.param_out}' — the base class injects it automatically."
            )
        otb_params[self.param_out] = self._format_otb_output(out_path, **params)

        cmd = self._build_cmd(cmd_name, otb_params)
        proc = subprocess.run(cmd, capture_output=False, text=True)  # noqa: S603
        if proc.returncode != 0:
            raise RuntimeError(
                f"OTB '{self.otb_app}' failed (exit {proc.returncode}):\n{proc.stderr}"
            )

        return StepOutput(outputs=[FlushedOutput(out_path)])

    def _format_otb_output(self, out_path: Path, **params: Any) -> str:
        """Return the OTB output path string, optionally with extended filename options.

        Override in subclasses to append GDAL creation options via OTB's extended
        filename syntax (``path.tif?&gdal:co:COMPRESS=DEFLATE&...``). The base
        implementation returns the plain path; ``FlushedOutput`` always tracks the
        clean path regardless of any suffix added here.
        """
        return str(out_path)

    def _build_cmd(self, cmd_name: str, otb_params: Dict[str, Any]) -> List[str]:
        """Flatten an OTB parameter dict into a CLI argument list.

        Boolean values are serialised as ``"true"``/``"false"``.
        List/tuple values are expanded as consecutive positional arguments
        (no repeated flag) — required by multi-input OTB parameters.
        """
        cmd: List[str] = [cmd_name]
        for key, value in otb_params.items():
            cmd.append(f"-{key}")
            if isinstance(value, bool):
                cmd.append("true" if value else "false")
            elif isinstance(value, (list, tuple)):
                cmd.extend(str(v) for v in value)
            else:
                cmd.append(str(value))
        return cmd

    @abstractmethod
    def build_otb_params(
        self,
        inputs: List[Path],
        output_path: Path,
        **params: Any,
    ) -> Dict[str, Any]:
        """Build the OTB parameter dictionary for one execution.

        Do **not** include the output key (``self.param_out``) — the base class
        injects it automatically after this method returns.

        Args:
            inputs: Inputs for this invocation (mirrors what ``execute`` received).
            output_path: Resolved output path for this invocation.
            **params: Step-specific parameters from the composition.

        Returns:
            Dict mapping OTB parameter names to their values.
        """
        ...
