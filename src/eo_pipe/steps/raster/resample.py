from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling

from eo_pipe.io.output_types import RasterOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.io.raster_io import DEFAULT_WRITER, RasterWriter
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

logger = setup_logger("eo_pipe.steps.resample")

_RESAMPLE_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "average": Resampling.average,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
}


def _compute_resample(
    input_file: Union[str, Path],
    downsample_factor: Optional[float] = None,
    target_resolution: Optional[float] = None,
    nodata_value: int = 0,
    method: str = "average",
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    """Read and resample a raster; return data, updated metadata, and source resolution.

    Does not write anything to disk — callers decide how to persist the result.

    Raises:
        ValueError: On invalid parameter combinations or zero-resolution input.
    """
    if method not in _RESAMPLE_METHODS:
        raise ValueError(
            f"Resampling method '{method}' not known. "
            f"Available: {list(_RESAMPLE_METHODS)}"
        )
    if downsample_factor is None and target_resolution is None:
        raise ValueError(
            "Either downsample_factor or target_resolution must be provided."
        )
    if downsample_factor is not None and target_resolution is not None:
        raise ValueError(
            "Provide either downsample_factor or target_resolution, not both."
        )

    with rio.open(input_file) as src:
        current_res: float = src.transform[0]
        if current_res == 0:
            raise ValueError(f"Input raster {input_file} has zero resolution.")

        factor = (
            target_resolution / current_res
            if target_resolution is not None
            else downsample_factor
        )
        assert factor is not None  # narrowed above

        new_height = max(1, int(round(src.height / factor)))
        new_width = max(1, int(round(src.width / factor)))

        data: np.ndarray = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=_RESAMPLE_METHODS[method],
        )

        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height,
        )

        metadata: Dict[str, Any] = src.meta.copy()
        metadata.update(
            {
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "nodata": nodata_value,
                "driver": "GTiff",
            }
        )

    return data, metadata, current_res


def downsample_raster(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    downsample_factor: Optional[float] = None,
    target_resolution: Optional[float] = None,
    nodata_value: int = 0,
    method: str = "average",
    writer: Optional[RasterWriter] = None,
) -> Path:
    """Resample a raster by factor or to a target resolution and write to disk.

    Either *downsample_factor* or *target_resolution* must be provided, not
    both.  Both up- and down-sampling are supported.

    Args:
        input_file: Path to the input raster.
        output_file: Path for the resampled output.
        downsample_factor: Explicit scale factor (``> 1`` = coarser,
                           ``< 1`` = finer).
        target_resolution: Target pixel size in the same units as the
                           raster's CRS.
        nodata_value: Nodata value to write in the output.
        method: Resampling algorithm name; one of ``nearest``,
                ``bilinear``, ``average``, ``cubic``, ``lanczos``.
        writer: Writer instance.  Defaults to :data:`DEFAULT_WRITER`.

    Returns:
        Resolved output path.

    Raises:
        ValueError: On invalid parameter combinations.
    """
    data, metadata, current_res = _compute_resample(
        input_file,
        downsample_factor=downsample_factor,
        target_resolution=target_resolution,
        nodata_value=nodata_value,
        method=method,
    )
    output_file = Path(output_file)
    (writer or DEFAULT_WRITER).write(output_file, data, **metadata)

    actual_res: float = metadata["transform"][0]
    logger.info(
        f"Resampled {Path(input_file).name} → {output_file.name} "
        f"({current_res:.4f} → {actual_res:.4f})"
    )
    return output_file


@StepRegistry.register
class ResampleStep(StepBase):
    """Resample each input raster to a target resolution.

    The write is deferred: :meth:`execute` returns :class:`~eo_pipe.io.output_types.RasterOutput`
    objects whose :meth:`~eo_pipe.io.output_types.RasterOutput.flush` is called by
    the batch strategy (or explicitly via :meth:`~eo_pipe.pipeline.base.StepOutput.flush_all`).

    Parameters (passed via ``**params`` in :meth:`execute`):
        target_resolution (float): Target pixel size in CRS units.
        method (str): Resampling algorithm; one of ``nearest``,
            ``bilinear``, ``average``, ``cubic``, ``lanczos``.
            Defaults to ``"average"``.
        nodata_value (int): Nodata value written to the output.
            Defaults to ``0``.
    """

    name = "resample"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        target_resolution = float(params["target_resolution"])
        method = str(params.get("method", "average"))
        nodata_value = int(params.get("nodata_value", 0))

        outputs = []
        for inp in inputs:
            out = self._path_strategy.resolve(self.name, inp, 0, output_dir)
            outputs.append(self._build_output(inp, out, target_resolution, method, nodata_value))

        return StepOutput(outputs=outputs)

    def _build_output(
        self,
        inp: Path,
        out: Path,
        target_resolution: float,
        method: str,
        nodata_value: int,
    ) -> RasterOutput:
        data, metadata, current_res = _compute_resample(
            inp,
            target_resolution=target_resolution,
            nodata_value=nodata_value,
            method=method,
        )
        actual_res: float = metadata["transform"][0]
        logger.info(
            f"Resampled {inp.name} → {out.name} "
            f"({current_res:.4f} → {actual_res:.4f})"
        )
        return RasterOutput(data=data, path=out, meta=metadata, writer=self._writer)
