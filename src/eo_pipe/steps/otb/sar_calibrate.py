"""SARCalibrationStep — OTB SARCalibration wrapped as a pipeline step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal

from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase

_VALID_LUTS: frozenset[str] = frozenset({"sigma", "gamma", "beta", "dn"})


@StepRegistry.register
class SARCalibrationStep(OTBStepBase):
    """Calibrate a raw S1 GRD image to physical backscatter values.

    Uses the OTB ``SARCalibration`` application to convert raw DN values to
    sigma naught (σ⁰), gamma naught (γ⁰), or beta naught (β⁰) using the
    calibration LUTs embedded in the S1 product metadata.

    Use :class:`~eo_pipe.pipeline.batch.ParallelBatch` to calibrate VV and VH
    polarisations independently::

        import eo_pipe
        from eo_pipe import PipelineComposition, ParallelBatch
        from pathlib import Path

        ctx = (
            PipelineComposition(workspace=Path("/data/work"))
            .add_step("sar_calibrate", ParallelBatch(),
                      lut="sigma", removenoise=False)
            .run(inputs=[Path("s1a-vv.tif"), Path("s1a-vh.tif")])
        )

    Parameters:
        lut (str):
            Output calibration type: ``"sigma"`` (σ⁰, default), ``"gamma"``
            (γ⁰), ``"beta"`` (β⁰), or ``"dn"`` (raw DN passthrough).
        removenoise (bool):
            Remove thermal noise using the LUT in S1 product metadata.
            Default: ``False``. Enable only for products with intact noise
            LUTs — corrupted LUTs cause a C++ segfault inside OTB.
        ram_mb (int):
            RAM budget in megabytes. Default: ``256``.
        compress (bool):
            Write DEFLATE-compressed tiled GeoTIFF. Default: ``True``.
    """

    name = "sar_calibrate"
    otb_app = "SARCalibration"

    def build_otb_params(
        self,
        inputs: List[Path],
        output_path: Path,
        **params: Any,
    ) -> Dict[str, Any]:
        """Build ``SARCalibration`` parameter dict.

        Expected keys in ``params``:
            lut (str): Calibration LUT name. Default: ``"sigma"``.
            removenoise (bool): Enable thermal noise removal. Default: ``False``.

        Returns:
            OTB parameter dict (without the output key).
        """
        if len(inputs) != 1:
            raise ValueError(
                f"SARCalibrationStep expects exactly one input per call; got {len(inputs)}. "
                "Use ParallelBatch to calibrate multiple files independently."
            )

        lut: str = params.get("lut", "sigma")
        if lut not in _VALID_LUTS:
            raise ValueError(
                f"Invalid lut {lut!r}. Valid values: {sorted(_VALID_LUTS)}"
            )

        removenoise: bool = bool(params.get("removenoise", False))
        ram_mb: int = int(params.get("ram_mb", 256))

        return {
            self.param_in: str(inputs[0]),
            "lut": lut,
            "removenoise": removenoise,
            "opt.ram": ram_mb,
        }
