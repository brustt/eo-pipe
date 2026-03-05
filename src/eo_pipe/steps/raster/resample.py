from pathlib import Path
from typing import List

from eo_pipe.io.raster_io import downsample_raster
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry


@StepRegistry.register
class ResampleStep(StepBase):
    """Resample each input raster to a target resolution.

    Parameters (passed via ``**params`` in :meth:`execute`):
        target_resolution (float): Target pixel size in CRS units.
        method (str): Resampling algorithm; one of ``nearest``,
            ``bilinear``, ``average``, ``cubic``, ``lanczos``.
            Defaults to ``"average"``.
        nodata_value (int): Nodata value written to the output.
            Defaults to ``0``.
    """

    name = "resample"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        target_resolution = float(params["target_resolution"])
        method = params.get("method", "average")
        nodata_value = params.get("nodata_value", 0)

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            downsample_raster(
                input_file=inp,
                output_file=out,
                target_resolution=target_resolution,
                nodata_value=nodata_value,
                method=method,
            )
            outputs.append(out)

        return StepResult(outputs=outputs)
