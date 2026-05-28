"""Generic raster I/O utilities — no domain-specific logic."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import rasterio as rio

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
        nodata: Nodata value written to the output.  ``None`` (default) leaves
            whatever nodata is already in the profile unchanged.  Set
            explicitly when the step introduces a specific sentinel value.
        cog: Convert the output to a Cloud-Optimised GeoTIFF after writing.
        cog_blocksize: Internal tile size for the COG.
        num_threads: Number of threads for COG conversion.
        overview_resampling: Resampling for COG overviews.
        extra: Extra rasterio creation options merged last (overrides above).

    Example::

        writer = RasterWriter(compress="deflate", cog=True)
        writer.write(Path("out.tif"), data, **profile)

    .. note::
        ``0`` is intentionally **not** the default nodata value — it is a
        valid reflectance/radiance value in most EO datasets.  Choose a
        domain-appropriate sentinel (``-9999``, ``65535``, ``np.nan`` for
        float) when you need one, or leave ``nodata=None`` to propagate the
        source raster's nodata through unchanged.
    """

    compress: Compression = "deflate"
    tiled: bool = True
    blockxsize: int = 512
    blockysize: int = 512
    bigtiff: Literal["auto", "yes", "no"] = "auto"
    predictor: Optional[int] = None
    nodata: Optional[float] = None
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
        gcps: Optional[list] = None,
        gcp_crs: Optional[Any] = None,
        **profile: Any,
    ) -> Path:
        """Write *data* to a GeoTIFF, optionally converting to COG.

        If :attr:`nodata` is set on this writer, it overrides whatever nodata
        value is in *profile*.

        When *gcps* is provided, ``crs`` and ``transform`` are stripped from
        *profile* (GCPs and affine transform are mutually exclusive in GeoTIFF).

        Args:
            output_file: Destination path.
            data: Array shaped ``(bands, height, width)``.
            gcps: Ground control points to embed (e.g. from a S1 annotation XML).
            gcp_crs: CRS of the GCP coordinates (required when *gcps* is set).
            **profile: Rasterio profile keyword arguments.

        Returns:
            Resolved output path.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        creation = self._creation_options(data.dtype)
        profile.update(creation)
        profile["driver"] = "GTiff"

        if self.nodata is not None:
            profile["nodata"] = self.nodata

        if gcps:
            profile.pop("crs", None)
            profile.pop("transform", None)

        with rio.open(output_file, "w", **profile) as dst:
            dst.write(data)
            if gcps:
                dst.gcps = (gcps, gcp_crs)

        if self.cog:
            self._to_cog(output_file, profile.get("nodata", 0))

        return output_file

    def _to_cog(self, path: Path, nodata: Any) -> None:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        cog_profile: dict[str, Any] = cog_profiles.get("deflate")  # type: ignore[no-untyped-call]
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
                overview_resampling=self.overview_resampling,  # type: ignore[arg-type]
            )


DEFAULT_WRITER = RasterWriter()


# ---------------------------------------------------------------------------
# GDAL creation options
# ---------------------------------------------------------------------------


def _add_gdal_options() -> Dict[str, str]:
    """Return standard GDAL creation options for compressed tiled GeoTIFFs."""
    return {
        "COMPRESS": "DEFLATE",
        "BIGTIFF": "YES",
        "NUM_THREADS": "ALL_CPUS",
        "TILED": "YES",
        "BLOCKXSIZE": "256",
        "BLOCKYSIZE": "256",
    }


# ---------------------------------------------------------------------------
# Raster reader
# ---------------------------------------------------------------------------


@dataclass
class RasterData:
    """Result of reading a raster via :class:`RasterReader`.

    Attributes:
        data: Array shaped ``(bands, height, width)``.  Nodata pixels contain
              their original value unless *fill_value* was set on the reader.
        valid_mask: Boolean array ``(height, width)``; ``True`` = valid pixel,
                    ``False`` = nodata.  Derived from rasterio's
                    :meth:`~rasterio.DatasetReader.dataset_mask`, which handles
                    nodata values, alpha bands, and mask bands uniformly.
        profile: Rasterio profile dict (copy of source metadata).
        path: Source file path.
    """

    data: np.ndarray
    valid_mask: np.ndarray
    profile: Dict[str, Any]
    path: Path

    @property
    def has_nodata(self) -> bool:
        """``True`` if any pixel is marked as nodata."""
        return not bool(self.valid_mask.all())


@dataclass
class RasterReader:
    """Reads raster data with consistent nodata handling via rasterio's pixel mask.

    Uses :meth:`~rasterio.DatasetReader.dataset_mask` so nodata regions are
    correctly identified regardless of whether nodata is encoded as a nodata
    value, an alpha band, or a dedicated mask band.

    Args:
        fill_value: Value substituted for nodata pixels in the returned
            :attr:`~RasterData.data` array.  ``None`` (default) preserves the
            original pixel values at nodata positions — use
            :attr:`~RasterData.valid_mask` to identify them.  Set to a safe
            sentinel (e.g. ``0``) only when downstream code cannot handle the
            original nodata value (e.g. a filter that must not read masked
            pixels at all).

    Example::

        reader = RasterReader()
        rdata = reader.read(Path("dem.tif"))

        # Process only valid pixels
        values = rdata.data[:, rdata.valid_mask]

        # Restore nodata positions after in-place processing
        result[~rdata.valid_mask] = rdata.data[~rdata.valid_mask]
    """

    fill_value: Optional[float] = None

    def read(self, path: Union[str, Path]) -> RasterData:
        """Read *path* and return data alongside a valid-pixel mask.

        Args:
            path: Raster file to read.

        Returns:
            :class:`RasterData` with ``data``, ``valid_mask``, ``profile``,
            and ``path``.
        """
        path = Path(path)
        with rio.open(path) as src:
            gdal_mask = src.dataset_mask()  # (H, W) uint8: 0=nodata, 255=valid
            data = src.read()              # (bands, H, W)
            profile = src.profile.copy()

        valid_mask = gdal_mask == 255  # bool, True = valid pixel

        if self.fill_value is not None and not valid_mask.all():
            data = data.copy()
            for b in range(data.shape[0]):
                data[b][~valid_mask] = self.fill_value

        return RasterData(data=data, valid_mask=valid_mask, profile=profile, path=path)


DEFAULT_READER = RasterReader()
