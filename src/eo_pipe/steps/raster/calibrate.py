from pathlib import Path
from typing import Any, Dict, List, Optional

from eo_pipe.io.raster_io import hist_match_worker
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.pipeline.base import StepBase, StepResult
from eo_pipe.pipeline.registry import StepRegistry


@StepRegistry.register
class HistogramCalibrationStep(StepBase):
    """Radiometric calibration via histogram matching.

    Parameters (passed via ``**params``):
        ref_path (str | Path): **Required.** Reference raster for histogram
            matching.
        match_proportion (float): Blending factor ``[0, 1]``.
            Defaults to ``1.0``.
        bands (str): Comma-separated 1-based band indices.
            Defaults to ``"1,2,3"``.
        color_space (str): Colour space: ``"RGB"``, ``"LAB"``, or ``"LCH"``.
            Defaults to ``"RGB"``.
        creation_options (dict): Rasterio creation options.
    """

    name = "calibrate"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params) -> StepResult:
        ref_path = Path(params["ref_path"])
        match_proportion: float = float(params.get("match_proportion", 1.0))
        bands: str = params.get("bands", "1,2,3")
        color_space: str = params.get("color_space", "RGB")
        creation_options: Optional[Dict[str, Any]] = params.get(
            "creation_options", {}
        )

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            hist_match_worker(
                src_path=inp,
                ref_path=ref_path,
                dst_path=out,
                match_proportion=match_proportion,
                creation_options=creation_options,
                bands=bands,
                color_space=color_space,
                save=True,
            )
            outputs.append(out)

        return StepResult(outputs=outputs)
