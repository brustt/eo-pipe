from pathlib import Path
from typing import List

from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry


@StepRegistry.register
class SieveStep(StepBase):
    """Remove small connected regions from a classified raster (GDAL sieve).

    Parameters (passed via ``**params``):
        threshold (int): Minimum region size in pixels to keep.
            Defaults to ``20``.
        connectedness (int): Pixel connectedness: ``4`` or ``8``.
            Defaults to ``8``.
    """

    name = "sieve"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        threshold = int(params.get("threshold", 20))
        connectedness = int(params.get("connectedness", 8))

        outputs = []
        from osgeo_utils.gdal_sieve import gdal_sieve  # lazy: requires system GDAL

        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            gdal_sieve(
                src_filename=str(inp),
                dst_filename=str(out),
                threshold=threshold,
                connectedness=connectedness,
            )
            outputs.append(out)

        return StepResult(outputs=outputs)
