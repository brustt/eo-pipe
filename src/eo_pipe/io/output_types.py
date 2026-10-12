"""Persistable output types for pipeline steps.

Steps return :class:`StepOutput` from :meth:`~eo_pipe.pipeline.base.StepBase.execute`.
Each item in ``StepOutput.outputs`` is a :class:`Persistable` — a pending write
that has not yet touched the filesystem.  The
:class:`~eo_pipe.pipeline.batch.BatchStrategy` calls :meth:`Persistable.flush`
on each item, collecting the resolved :class:`~pathlib.Path` objects that become
the next step's inputs.

Three concrete Persistable implementations cover all current step patterns:

* :class:`RasterOutput` — holds a NumPy array in memory; writes via a
  :class:`~eo_pipe.io.raster_io.RasterWriter` on flush.

* :class:`VectorOutput` — holds a GeoDataFrame in memory; writes via a
  :class:`VectorFormat` strategy on flush.

* :class:`FlushedOutput` — wraps a path already on disk; flush is a no-op.

Three built-in :class:`VectorFormat` strategies:

* :class:`GpkgFormat` — GeoPackage (``.gpkg``), the default.
* :class:`ShapefileFormat` — ESRI Shapefile (``.shp``).
* :class:`ParquetFormat` — GeoParquet (``.parquet``), requires geopandas ≥ 0.12.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, TYPE_CHECKING, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import geopandas as gpd
import rasterio as rio

if TYPE_CHECKING:
    import xarray as xr

    from eo_pipe.io.raster_io import RasterWriter


# ---------------------------------------------------------------------------
# Vector format strategies
# ---------------------------------------------------------------------------


class VectorFormat(ABC):
    """Strategy for writing a :class:`~geopandas.GeoDataFrame` to disk.

    Subclass this to add support for new formats without modifying any
    existing step code::

        class MyFormat(VectorFormat):
            extension = ".custom"
            def write(self, gdf, path):
                ...
                return path
    """

    #: File extension including the leading dot (e.g. ``".gpkg"``).
    extension: str

    @abstractmethod
    def write(self, gdf: "gpd.GeoDataFrame", path: Path) -> Path:
        """Write *gdf* to *path* and return the resolved path.

        Implementations must create parent directories if needed.
        """
        ...


class GpkgFormat(VectorFormat):
    """GeoPackage — the default vector format.

    Single-file, no auxiliary sidecar files, supports multiple layers.
    """

    extension = ".gpkg"

    def write(self, gdf: "gpd.GeoDataFrame", path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="GPKG")
        return path


class ShapefileFormat(VectorFormat):
    """ESRI Shapefile.

    Produces multiple sidecar files (``.dbf``, ``.shx``, …) alongside
    the ``.shp``.  Prefer :class:`GpkgFormat` for new workflows.
    """

    extension = ".shp"

    def write(self, gdf: "gpd.GeoDataFrame", path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="ESRI Shapefile")
        return path


class ParquetFormat(VectorFormat):
    """GeoParquet — columnar storage for large vector datasets.

    Requires geopandas ≥ 0.12 and ``pyarrow``.  Not supported by all GIS
    tools, but ideal for data-science / analytics pipelines.
    """

    extension = ".parquet"

    def write(self, gdf: "gpd.GeoDataFrame", path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(path)
        return path


# ---------------------------------------------------------------------------
# Persistable protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Persistable(Protocol):
    """IO contract for a pending step output.

    Calling :meth:`flush` writes the output to disk (or is a no-op if the
    file is already there) and returns the resolved path.

    .. note::
        ``path`` must be set at construction time so that callers can inspect
        the intended destination before flushing.
    """

    path: Path

    def flush(self) -> Path:
        """Write this output to disk and return its path."""
        ...


# ---------------------------------------------------------------------------
# Concrete Persistable types
# ---------------------------------------------------------------------------


@dataclass
class RasterOutput:
    """In-memory raster array that writes via a :class:`RasterWriter` on flush.

    The array is held in memory until :meth:`flush` is called by the
    :class:`~eo_pipe.pipeline.batch.BatchStrategy`.  Use
    :class:`FlushedOutput` for steps that must write incrementally (windowed
    readers, GDAL utilities).

    Args:
        data: Array shaped ``(bands, height, width)``.
        path: Intended output path.
        meta: Rasterio profile keyword arguments forwarded to the writer.
        writer: Writer instance injected by the step from ``self._writer``.
        tags: Dataset-level GeoTIFF tags written after the raster.
        descriptions: Per-band descriptions written after the raster.
    """

    data: np.ndarray
    path: Path
    meta: dict[str, Any]
    writer: "RasterWriter"
    tags: Optional[dict[str, str]] = None
    descriptions: Optional[Sequence[Optional[str]]] = None

    def flush(self) -> Path:
        """Write *data* to *path* via *writer* and return the resolved path."""
        path = self.writer.write(self.path, self.data, **self.meta)
        if self.tags or self.descriptions:
            with rio.open(path, "r+") as dst:
                if self.tags:
                    dst.update_tags(**self.tags)
                if self.descriptions:
                    dst.descriptions = tuple(self.descriptions)
        return path


@dataclass
class WindowedRasterOutput:
    """Raster produced and written window by window — the full array is never in memory.

    The file is written under a temporary name and renamed on success, so a failed
    write leaves nothing at *path*.

    Args:
        windows: Windows covering the raster, written in order.
        open_producer: Context-manager factory yielding a callable that maps a window
            to its ``(bands, rows, cols)`` array; it owns the source handles.
        path: Intended output path.
        meta: Rasterio profile keyword arguments forwarded to the writer.
        writer: Writer instance injected by the step from ``self._writer``.
        tags: Dataset-level GeoTIFF tags written after the raster.
        descriptions: Per-band descriptions written after the raster.
    """

    windows: Sequence[Any]
    open_producer: Callable[[], ContextManager[Callable[[Any], np.ndarray]]]
    path: Path
    meta: dict[str, Any]
    writer: "RasterWriter"
    tags: Optional[dict[str, str]] = None
    descriptions: Optional[Sequence[Optional[str]]] = None

    def flush(self) -> Path:
        """Stream the windows to *path* via *writer* and return the resolved path."""
        tmp = self.path.with_name(self.path.name + ".part")
        try:
            with self.open_producer() as produce:
                self.writer.write_windows(tmp, self.windows, produce, **self.meta)
            if self.tags or self.descriptions:
                with rio.open(tmp, "r+") as dst:
                    if self.tags:
                        dst.update_tags(**self.tags)
                    if self.descriptions:
                        dst.descriptions = tuple(self.descriptions)
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return self.path


@dataclass
class VectorOutput:
    """In-memory GeoDataFrame that writes via a :class:`VectorFormat` on flush.

    The format strategy controls both the file extension and the write
    implementation.  Use :class:`GpkgFormat` (the default) unless a specific
    format is required.

    Args:
        gdf: The GeoDataFrame to persist.
        path: Intended output path — must include the correct extension for
              the chosen format (use ``fmt.extension`` to construct it).
        fmt: Format strategy.  Defaults to :class:`GpkgFormat`.

    Example::

        fmt = ParquetFormat()
        out = VectorOutput(
            gdf=my_gdf,
            path=output_dir / f"result{fmt.extension}",
            fmt=fmt,
        )
        path = out.flush()
    """

    gdf: "gpd.GeoDataFrame"
    path: Path
    fmt: VectorFormat = field(default_factory=GpkgFormat)

    def flush(self) -> Path:
        """Write *gdf* to *path* via *fmt* and return the resolved path."""
        return self.fmt.write(self.gdf, self.path)


@dataclass
class FlushedOutput:
    """Wraps a path that is already written to disk — flush is a no-op.

    Used by streaming steps (e.g.
    :class:`~eo_pipe.steps.raster.resample.ResampleStep`) where IO happens
    inside an internal function.  :meth:`flush` returns the existing path,
    preserving the uniform :class:`Persistable` interface without re-writing
    anything.

    Args:
        path: Path of the already-written file.
    """

    path: Path

    def flush(self) -> Path:
        """Return the path — the file is already on disk."""
        return self.path


@dataclass
class XarrayOutput:
    """``xarray.Dataset`` that writes a NetCDF file on flush.

    The Dataset may be in memory or dask-backed.  A dask-backed Dataset is computed
    and streamed to disk chunk by chunk at flush time, with on-disk chunking matching
    its dask chunks (so a full time axis per chunk stays one on-disk chunk).  The file
    is written under a temporary name and renamed on success, so a failed write never
    leaves a valid-looking file at *path*.

    Args:
        ds: The Dataset to persist.
        path: Intended output path (``.nc``).
        num_workers: Chunks computed concurrently on the local threaded scheduler
            (dask default when ``None``).
    """

    ds: "xr.Dataset"
    path: Path
    num_workers: Optional[int] = None

    def _with_chunk_encoding(self) -> "xr.Dataset":
        ds = self.ds
        for name, var in ds.data_vars.items():
            if var.chunks is not None and "chunksizes" not in var.encoding:
                ds[name].encoding["chunksizes"] = tuple(c[0] for c in var.chunks)
        return ds

    def flush(self) -> Path:
        """Write *ds* to *path* as NetCDF4 and return the path."""
        import dask

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        try:
            config: dict[str, Any] = {"scheduler": "threads"}
            if self.num_workers:
                config["num_workers"] = self.num_workers
            with dask.config.set(config):
                self._with_chunk_encoding().to_netcdf(tmp, engine="netcdf4")
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return self.path
