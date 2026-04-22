from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rasterio as rio
from rasterio.transform import guard_transform

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.io.raster_io import DEFAULT_WRITER, RasterWriter
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

logger = setup_logger("eo_pipe.steps.calibrate")


# ---------------------------------------------------------------------------
# Histogram matching internals
# ---------------------------------------------------------------------------


def _build_lut(src_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
    """Build an integer look-up table via ``np.bincount``.

    O(n) time and O(max_value) memory.  Works for uint8 (256 buckets) and
    uint16 (65 536 buckets).  *src_data* and *ref_data* must be 1-D and share
    the same integer dtype.

    Args:
        src_data: Flat source values (valid pixels only).
        ref_data: Flat reference values.

    Returns:
        1-D LUT of the same dtype: ``lut[v]`` gives the matched value for
        source value ``v``.
    """
    n = int(np.iinfo(src_data.dtype).max) + 1
    src_cdf = np.bincount(src_data, minlength=n).cumsum()
    ref_cdf = np.bincount(ref_data, minlength=n).cumsum()
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    lut = np.searchsorted(ref_cdf, src_cdf).clip(0, n - 1).astype(src_data.dtype)
    return lut


def _match_band_integer(
    src_band: np.ndarray,
    ref_band: np.ndarray,
    valid_mask: Optional[np.ndarray],
    match_proportion: float,
) -> np.ndarray:
    """Histogram match one integer band via LUT.  O(n), no float allocation."""
    src_data = src_band[valid_mask].ravel() if valid_mask is not None else src_band.ravel()
    lut = _build_lut(src_data, ref_band.ravel())

    if match_proportion == 1.0:
        return lut[src_band]

    matched_f = lut[src_band].astype(np.float32)
    blended = src_band.astype(np.float32) * (1.0 - match_proportion) + matched_f * match_proportion
    hi = np.iinfo(src_band.dtype).max
    return np.clip(blended, 0, hi).astype(src_band.dtype)


def _match_band_float(
    src_band: np.ndarray,
    ref_band: np.ndarray,
    match_proportion: float,
) -> np.ndarray:
    """Histogram match one float band via ``skimage.exposure.match_histograms``."""
    from skimage.exposure import match_histograms
    matched = match_histograms(src_band, ref_band).astype(src_band.dtype)
    if match_proportion != 1.0:
        matched = (src_band * (1.0 - match_proportion) + matched * match_proportion).astype(
            src_band.dtype
        )
    return matched


def _cs_forward(arr: np.ndarray, color_space: str) -> np.ndarray:
    """Normalise *arr* (bands-first, integer or float) to the given colour space.

    Uses float32 throughout to halve memory vs float64.  Only the first three
    bands are used.
    """
    cs = color_space.upper()
    dtype_max = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
    f32 = arr[:3].astype(np.float32) / dtype_max

    if cs == "RGB":
        return f32
    if cs == "LAB":
        from skimage.color import rgb2lab
        return rgb2lab(f32.transpose(1, 2, 0)).transpose(2, 0, 1).astype(np.float32)
    if cs == "LCH":
        from skimage.color import rgb2lab
        lab = rgb2lab(f32.transpose(1, 2, 0)).astype(np.float32)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.sqrt(a**2 + b**2)
        H = np.arctan2(b, a)
        return np.stack([L, C, H], axis=0)
    raise ValueError(f"Unsupported colour space: '{color_space}'")


def _cs_backward(arr: np.ndarray, color_space: str) -> np.ndarray:
    """Convert bands-first float32 *arr* back to uint8 RGB."""
    cs = color_space.upper()
    if cs == "RGB":
        return np.clip(arr * 255, 0, 255).astype(np.uint8)
    if cs == "LAB":
        from skimage.color import lab2rgb
        rgb = lab2rgb(arr.transpose(1, 2, 0)).transpose(2, 0, 1)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
    if cs == "LCH":
        from skimage.color import lab2rgb
        L, C, H = arr[0], arr[1], arr[2]
        a = C * np.cos(H)
        b = C * np.sin(H)
        lab = np.stack([L, a, b], axis=-1)
        rgb = lab2rgb(lab).transpose(2, 0, 1)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported colour space: '{color_space}'")


def hist_match_worker(
    src_path: Union[str, Path],
    ref_path: Union[str, Path],
    dst_path: Union[str, Path],
    match_proportion: float = 1.0,
    creation_options: Optional[Dict[str, Any]] = None,
    bands: str = "1,2,3",
    color_space: str = "RGB",
    save: bool = True,
    writer: Optional[RasterWriter] = None,
) -> Path:
    """Match the histogram of *src_path* to *ref_path* and write to *dst_path*.

    **Fast path (``color_space="RGB"`` with integer source)**:
    Uses an O(n) LUT built from ``np.bincount`` — no float64 allocation,
    no ``np.unique``.  Only valid (non-masked) source pixels contribute to
    the CDF so nodata regions do not skew the result.

    **Float / colour-space path (LAB, LCH)**:
    Converts to float32 (not float64) then delegates per-band matching to
    ``skimage.exposure.match_histograms``.

    Args:
        src_path: Source raster path.
        ref_path: Reference raster path.
        dst_path: Destination raster path.
        match_proportion: Blending between source (0) and full match (1).
        creation_options: Rasterio creation options merged into the profile.
        bands: Comma-separated 1-based band indices to match.
        color_space: ``"RGB"`` (default), ``"LAB"``, or ``"LCH"``.
        save: If ``False``, skip writing the output file.
        writer: Writer instance.  Defaults to :data:`DEFAULT_WRITER`.

    Returns:
        Resolved destination path.
    """
    creation_options = creation_options or {}
    dst_path = Path(dst_path)

    logger.info(
        f"Histogram matching {Path(src_path).name} → {Path(ref_path).name} "
        f"(space={color_space})"
    )

    with rio.open(src_path) as src:
        profile = src.profile.copy()
        src_arr = src.read()
        gdal_mask = src.dataset_mask()

    with rio.open(ref_path) as ref:
        ref_arr = ref.read()

    bixs = tuple(int(x) - 1 for x in bands.split(","))
    valid_mask = gdal_mask == 255
    has_nodata = not valid_mask.all()

    cs = color_space.upper()

    if cs == "RGB":
        target = src_arr.copy()
        is_integer = np.issubdtype(src_arr.dtype, np.integer)

        for b in bixs:
            if is_integer:
                target[b] = _match_band_integer(
                    src_arr[b], ref_arr[b], valid_mask if has_nodata else None, match_proportion
                )
            else:
                target[b] = _match_band_float(src_arr[b], ref_arr[b], match_proportion)

        out_dtype = src_arr.dtype

    else:
        src_cs = _cs_forward(src_arr, cs)
        ref_cs = _cs_forward(ref_arr, cs)

        target_cs = src_cs.copy()
        for b in bixs:
            target_cs[b] = _match_band_float(src_cs[b], ref_cs[b], match_proportion)

        target = _cs_backward(target_cs, cs)
        out_dtype = np.dtype("uint8")

    if has_nodata:
        for b in range(target.shape[0]):
            target[b][~valid_mask] = src_arr[b][~valid_mask]

    profile.update(
        {
            "dtype": np.dtype(out_dtype).name,
            "count": target.shape[0],
            "transform": guard_transform(profile["transform"]),
        }
    )
    profile.update(creation_options)

    if save:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        (writer or DEFAULT_WRITER).write(dst_path, target.astype(out_dtype), **profile)
        if has_nodata:
            with rio.open(dst_path, "r+") as dst:
                dst.write_mask(gdal_mask)

    return dst_path


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path_strategy = PrefixedPathStrategy()

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        ref_path = Path(params["ref_path"])
        match_proportion: float = float(params.get("match_proportion", 1.0))
        bands: str = params.get("bands", "1,2,3")
        color_space: str = params.get("color_space", "RGB")
        creation_options: Optional[Dict[str, Any]] = params.get("creation_options", {})

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
                writer=self._writer,
            )
            outputs.append(FlushedOutput(out))

        return StepOutput(outputs=outputs)
