"""Abstract base for OTB-backed pipeline steps."""

from __future__ import annotations

import logging
import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List

import rasterio

from eo_pipe.io.output_types import FlushedOutput

logger = logging.getLogger(__name__)
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.io.raster_io import _add_gdal_options
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
    _apply_gdal_options: ClassVar[bool] = True

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
        # TODO: implement production: capture OTB output : subprocess.PIPE
        proc = subprocess.run(cmd)  # noqa: S603
        if proc.returncode != 0:
            raise RuntimeError(
                f"OTB '{self.otb_app}' failed (exit {proc.returncode}):\n{proc.stdout}"
            )

        self._restore_crs(inputs[0], out_path)
        return StepOutput(outputs=[FlushedOutput(out_path)])

    def _restore_crs(self, src: Path, dst: Path) -> None:
        """Copy georeferencing from *src* to *dst* when OTB drops it.

        Handles three cases: CRS+transform, transform-only (S1 raw TIFFs with
        no embedded projection tag), and GCP-based georeferencing.
        """
        if not dst.exists():
            return
        with rasterio.open(dst) as ds:
            if ds.crs is not None and not ds.transform.is_identity:
                return
        with rasterio.open(src) as src_ds:
            crs = src_ds.crs
            transform = src_ds.transform
            gcps, gcp_crs = src_ds.gcps

        has_transform = not transform.is_identity
        if crs is None and not has_transform and not gcps:
            return

        with rasterio.open(dst, "r+") as dst_ds:
            if crs is not None:
                dst_ds.crs = crs
            if has_transform:
                dst_ds.transform = transform
            if gcps:
                dst_ds.gcps = (gcps, gcp_crs)
        logger.info("Restored georeferencing from %s to OTB output %s", src.name, dst.name)

    def _format_otb_output(self, out_path: Path, **params: Any) -> str:
        """Return the OTB output path string, with GDAL creation options appended when supported."""
        if not self._apply_gdal_options or not params.get("compress", True):
            return str(out_path)
        opts = _add_gdal_options()
        suffix = "?&" + "&".join(f"gdal:co:{k}={v}" for k, v in opts.items())
        return f"{out_path}{suffix}"

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
