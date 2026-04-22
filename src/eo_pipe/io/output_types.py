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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import geopandas as gpd

if TYPE_CHECKING:
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
    """

    data: np.ndarray
    path: Path
    meta: dict[str, Any]
    writer: "RasterWriter"

    def flush(self) -> Path:
        """Write *data* to *path* via *writer* and return the resolved path."""
        return self.writer.write(self.path, self.data, **self.meta)


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
