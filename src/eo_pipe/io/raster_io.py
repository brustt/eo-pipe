"""Generic raster I/O utilities — no domain-specific logic."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import guard_transform

from eo_pipe.logging import setup_logger

logger = setup_logger("eo_pipe.io.raster")


# ---------------------------------------------------------------------------
# Raster writer
# ---------------------------------------------------------------------------

Compression = Literal["deflate", "lzw", "zstd", "lzma", "none"]


@dataclass
class RasterWriter:
    """Configurable GeoTIFF writer with compression, tiling, BigTIFF, and COG.

    Args:
        compress: Compression algorithm.  ``"none"`` disables compression.
        tiled: Write tiled GeoTIFF.
        blockxsize: Tile width in pixels.
        blockysize: Tile height in pixels.
        bigtiff: BigTIFF handling.  ``"auto"`` lets GDAL decide (>4 GB),
            ``"yes"`` forces BigTIFF, ``"no"`` forces classic TIFF.
        predictor: Compression predictor.  ``1`` = none, ``2`` = horizontal
            differencing (integers), ``3`` = floating-point predictor.
            ``None`` auto-selects based on dtype (``2`` for int, ``3`` for float).
        cog: Convert the output to a Cloud-Optimised GeoTIFF after writing.
        cog_blocksize: Internal tile size for the COG.
        num_threads: Number of threads for COG conversion.
        overview_resampling: Resampling for COG overviews.
        extra: Extra rasterio creation options merged last (overrides above).

    Example::

        writer = RasterWriter(compress="deflate", cog=True)
        writer.write(Path("out.tif"), data, **profile)
    """

    compress: Compression = "deflate"
    tiled: bool = True
    blockxsize: int = 512
    blockysize: int = 512
    bigtiff: Literal["auto", "yes", "no"] = "auto"
    predictor: Optional[int] = None
    cog: bool = False
    cog_blocksize: int = 512
    num_threads: int = 4
    overview_resampling: str = "nearest"
    extra: Dict[str, Any] = field(default_factory=dict)

    def _resolve_predictor(self, dtype: np.dtype) -> int:
        if self.predictor is not None:
            return self.predictor
        if np.issubdtype(dtype, np.floating):
            return 3
        return 2

    def _creation_options(self, dtype: np.dtype) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "driver": "GTiff",
            "tiled": self.tiled,
        }
        if self.tiled:
            opts["blockxsize"] = self.blockxsize
            opts["blockysize"] = self.blockysize

        bigtiff_val = self.bigtiff.upper()
        if bigtiff_val == "AUTO":
            opts["BIGTIFF"] = "IF_SAFER"
        elif bigtiff_val == "YES":
            opts["BIGTIFF"] = "YES"

        if self.compress != "none":
            opts["compress"] = self.compress
            opts["predictor"] = self._resolve_predictor(dtype)

        opts.update(self.extra)
        return opts

    def write(
        self,
        output_file: Union[str, Path],
        data: np.ndarray,
        **profile,
    ) -> Path:
        """Write *data* to a GeoTIFF, optionally converting to COG.

        Args:
            output_file: Destination path.
            data: Array shaped ``(bands, height, width)``.
            **profile: Rasterio profile keyword arguments.

        Returns:
            Resolved output path.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        creation = self._creation_options(data.dtype)
        profile.update(creation)
        profile["driver"] = "GTiff"

        with rio.open(output_file, "w", **profile) as dst:
            dst.write(data)

        if self.cog:
            self._to_cog(output_file, profile.get("nodata", 0))

        return output_file

    def _to_cog(self, path: Path, nodata: Any) -> None:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        cog_profile = cog_profiles.get("deflate")
        cog_profile.update({
            "blocksize": self.cog_blocksize,
            "predictor": 2,
        })
        with rio.open(path) as src_ds:
            cog_translate(
                src_ds,
                path,
                cog_profile,
                nodata=nodata,
                in_memory=False,
                overview_resampling=self.overview_resampling,
                threads=self.num_threads,
            )


DEFAULT_WRITER = RasterWriter()


# ---------------------------------------------------------------------------
# Resample / downsample
# ---------------------------------------------------------------------------

_RESAMPLE_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "average": Resampling.average,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
}


def downsample_raster(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    downsample_factor: Optional[int] = None,
    target_resolution: Optional[float] = None,
    nodata_value: int = 0,
    method: str = "average",
) -> Path:
    """Resample a raster by factor or to a target resolution.

    Either *downsample_factor* or *target_resolution* must be provided, not
    both.  Unlike the previous implementation, both up- and down-sampling are
    supported.

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

    Returns:
        Resolved output path.

    Raises:
        ValueError: On invalid parameter combinations.
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
        current_res = src.transform[0]
        if current_res == 0:
            raise ValueError(f"Input raster {input_file} has zero resolution.")

        if target_resolution is not None:
            downsample_factor = target_resolution / current_res

        new_height = max(1, int(round(src.height / downsample_factor)))
        new_width = max(1, int(round(src.width / downsample_factor)))
        new_shape = (src.count, new_height, new_width)

        data = src.read(
            out_shape=new_shape, resampling=_RESAMPLE_METHODS[method]
        )

        scale_x = src.width / new_width
        scale_y = src.height / new_height
        new_transform = src.transform * src.transform.scale(scale_x, scale_y)

        metadata = src.meta.copy()
        metadata.update(
            {
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "nodata": nodata_value,
                "driver": "GTiff",
            }
        )

    output_file = Path(output_file)
    DEFAULT_WRITER.write(output_file, data, **metadata)

    actual_res = metadata["transform"][0]
    logger.info(
        f"Resampled {Path(input_file).name} → {output_file.name} "
        f"({current_res:.4f} → {actual_res:.4f})"
    )
    return output_file


# ---------------------------------------------------------------------------
# Histogram matching
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
    # Normalise to [0, 1]
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    # For each source percentile find the reference value with the closest CDF
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

    # Blend in float32 then cast back to original dtype
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
    f32 = arr[:3].astype(np.float32) / dtype_max  # float32, not float64

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
        src_arr = src.read()          # (bands, H, W) — original dtype, no masked overhead
        gdal_mask = src.dataset_mask()  # 0 = nodata, 255 = valid

    with rio.open(ref_path) as ref:
        ref_arr = ref.read()

    bixs = tuple(int(x) - 1 for x in bands.split(","))
    valid_mask = gdal_mask == 255   # 2-D bool; True = valid pixel
    has_nodata = not valid_mask.all()

    cs = color_space.upper()

    if cs == "RGB":
        # -----------------------------------------------------------
        # Fast path: work directly on the original dtype (uint8/uint16)
        # No colour-space conversion, no large float64 intermediaries.
        # -----------------------------------------------------------
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
        # -----------------------------------------------------------
        # Colour-space path (LAB / LCH): float32, then skimage
        # -----------------------------------------------------------
        src_cs = _cs_forward(src_arr, cs)
        ref_cs = _cs_forward(ref_arr, cs)

        target_cs = src_cs.copy()
        for b in bixs:
            target_cs[b] = _match_band_float(src_cs[b], ref_cs[b], match_proportion)

        target = _cs_backward(target_cs, cs)   # uint8
        out_dtype = np.dtype("uint8")

    # Restore nodata pixels to their original values
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
        DEFAULT_WRITER.write(dst_path, target.astype(out_dtype), **profile)
        if has_nodata:
            with rio.open(dst_path, "r+") as dst:
                dst.write_mask(gdal_mask)

    return dst_path


# ---------------------------------------------------------------------------
# Hillshade
# ---------------------------------------------------------------------------


def create_hillshade(
    dem_path: Union[str, Path],
    output_path: Union[str, Path],
    z_factor: float = 1.0,
    azimuth: float = 300.0,
    altitude: float = 45.0,
) -> Path:
    """Create a hillshade from a DEM raster using GDAL DEMProcessing.

    Args:
        dem_path: Input DEM raster path.
        output_path: Destination path for the hillshade.
        z_factor: Vertical exaggeration factor.
        azimuth: Light source azimuth angle in degrees.
        altitude: Light source altitude angle in degrees.

    Returns:
        Resolved output path.

    Raises:
        IOError: If the DEM file cannot be opened.
    """
    from osgeo import gdal  # lazy: osgeo requires system GDAL

    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dem_dataset = gdal.Open(str(dem_path))
    if not dem_dataset:
        raise IOError(f"Could not open DEM file: {dem_path}")

    options = gdal.DEMProcessingOptions(
        format="GTiff",
        zFactor=z_factor,
        azimuth=azimuth,
        altitude=altitude,
    )
    gdal.DEMProcessing(
        destName=str(output_path),
        srcDS=dem_dataset,
        processing="hillshade",
        options=options,
    )
    dem_dataset = None  # Close GDAL dataset

    logger.info(f"Hillshade written to {output_path}")
    return output_path

