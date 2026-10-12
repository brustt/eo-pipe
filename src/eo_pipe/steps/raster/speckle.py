"""Out-of-core Sentinel-1 speckle filters: Lee (spatial) and Quegan (multi-temporal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional

import dask.array as da
import numpy as np
import rasterio as rio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from eo_pipe.io.memory import budget_mb
from eo_pipe.io.output_types import DaskRasterOutput, XarrayOutput
from eo_pipe.io.path_utils import PrefixedPathStrategy
from eo_pipe.io.speckle import (
    DEFAULT_ENL,
    DEFAULT_WINDOW,
    check_window,
    filter_lazily,
    lee_block,
    plan_filter_chunks,
    quegan_block,
    require_linear,
)
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

_COMPRESSION = {"zlib": True, "complevel": 1, "shuffle": True}
_NO_COMPRESSION = {"zlib": False, "shuffle": False}
_STALE_ENCODING = ("chunksizes", "preferred_chunks", "contiguous", "zlib", "complevel", "shuffle", "szip", "zstd")


class _SpeckleFilterStep(StepBase):
    """Shared plumbing: open lazily, chunk by memory budget, filter with a halo, persist lazily."""

    #: Kernel applied to every ``(..., rows, cols)`` block.
    kernel: ClassVar[Callable[..., np.ndarray]]
    #: ``.tif`` input is accepted (a Tile stack ``.nc`` always is).
    accepts_tif: ClassVar[bool] = True
    #: Fewest dates a Tile stack must hold.
    min_dates: ClassVar[int] = 1

    def kernel_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        window = check_window(params.get("window_size", DEFAULT_WINDOW))
        budget = budget_mb(params)
        compress = bool(params.get("compress", False))
        kernel_kwargs = {"window": window, **self.kernel_params(params)}
        strategy = PrefixedPathStrategy()
        outputs: List[Any] = []
        for inp in map(Path, inputs):
            out = strategy.resolve(self.name, inp, 0, output_dir)
            suffix = inp.suffix.lower()
            if suffix == ".nc":
                outputs.append(self._filter_netcdf(inp, out, budget, compress, kernel_kwargs))
            elif suffix in (".tif", ".tiff"):
                if not self.accepts_tif:
                    raise ValueError(f"{self.name}: {inp.name} is a GeoTIFF; a Tile stack (.nc) is required.")
                outputs.append(self._filter_geotiff(inp, out, budget, kernel_kwargs))
            else:
                raise ValueError(f"{self.name}: unsupported input {inp.name}; expected .tif or .nc.")
        return StepOutput(outputs=outputs)

    def _plan(self, name: str, dates: int, height: int, width: int, budget: float, kw: Dict[str, Any]) -> Any:
        plan = plan_filter_chunks(budget, dates, height, width, kw["window"])
        self._logger.info(
            "%s: %d date(s), %dx%d px, chunks of %d px (+%d px halo), %d at a time",
            name, dates, width, height, plan.side, kw["window"] // 2, plan.workers,
        )
        return plan

    def _filter_netcdf(self, inp: Path, out: Path, budget: float, compress: bool, kw: Dict[str, Any]) -> XarrayOutput:
        ds = xr.open_dataset(inp, chunks={})
        if not ds.data_vars:
            raise ValueError(f"{inp.name}: no data variables to filter.")
        filtered = ds.copy()
        workers = None
        for name, var in ds.data_vars.items():
            if var.ndim < 2:
                raise ValueError(f"{inp.name}: variable '{name}' is not a raster.")
            require_linear({**ds.attrs, **var.attrs}, f"{inp.name}:{name}")
            dates = int(np.prod(var.shape[:-2])) if var.ndim > 2 else 1
            if dates < self.min_dates:
                raise ValueError(
                    f"{self.name}: {inp.name}:{name} has {dates} date(s); at least {self.min_dates} are required."
                )
            plan = self._plan(f"{inp.name}:{name}", dates, var.shape[-2], var.shape[-1], budget, kw)
            workers = plan.workers if workers is None else min(workers, plan.workers)
            chunks = {d: -1 for d in var.dims[:-2]} | {var.dims[-2]: plan.side, var.dims[-1]: plan.side}
            nodata = var.attrs.get("nodata")
            result = filter_lazily(
                var.chunk(chunks).data, type(self).kernel, kw["window"] // 2,
                nodata=None if nodata is None else float(nodata), **kw,
            )
            filtered[name] = var.copy(data=result)
            enc = {k: v for k, v in var.encoding.items() if k not in _STALE_ENCODING}
            if np.dtype(result.dtype) != var.dtype and "_FillValue" in enc and enc["_FillValue"] is not None:
                enc["_FillValue"] = np.nan if np.dtype(result.dtype).kind == "f" else enc["_FillValue"]
            enc["dtype"] = str(result.dtype)
            filtered[name].encoding = enc
        return XarrayOutput(
            ds=filtered, path=out, num_workers=workers,
            encoding=dict(_COMPRESSION if compress else _NO_COMPRESSION),
        )

    def _filter_geotiff(self, inp: Path, out: Path, budget: float, kw: Dict[str, Any]) -> DaskRasterOutput:
        with rio.open(inp) as src:
            require_linear({"units": next(iter(u for u in src.units if u), "") or src.tags().get("units", "")},
                           inp.name)
            nodata = src.nodata
            count, height, width = src.count, src.height, src.width
            dtype = np.dtype(src.dtypes[0]) if np.dtype(src.dtypes[0]).kind == "f" else np.dtype("float32")
            meta = dict(
                dtype=dtype.name, count=count, width=width, height=height,
                crs=src.crs, transform=src.transform, nodata=nodata,
            )
            tags, descriptions = dict(src.tags()), list(src.descriptions)
        if count < self.min_dates:
            raise ValueError(f"{self.name}: {inp.name} has {count} band(s); at least {self.min_dates} are required.")
        plan = self._plan(inp.name, count, height, width, budget, kw)
        raster = rioxarray.open_rasterio(
            inp, chunks={"band": -1, "y": plan.side, "x": plan.side}, mask_and_scale=False, lock=True
        )
        assert isinstance(raster, xr.DataArray)
        result = filter_lazily(raster.data, type(self).kernel, kw["window"] // 2, nodata=nodata, **kw)
        if nodata is not None and not np.isnan(nodata):
            result = da.where(da.isnan(result), result.dtype.type(nodata), result)
        return DaskRasterOutput(
            data=result.astype(dtype), path=out, meta=meta, writer=self._writer,
            num_workers=plan.workers, tags=tags, descriptions=descriptions,
        )


@StepRegistry.register
class LeeFilterStep(_SpeckleFilterStep):
    """Classic Lee speckle filter, out-of-core.

    Accepts a ``.tif`` or a ``.nc`` Tile stack; every date and every variable (band) is
    filtered independently.  Input must be linear power (a ``units`` attribute of dB is
    refused).  Invalid pixels (nodata or NaN) are left out of the local statistics and
    restored as nodata in the output.  Output format and metadata follow the input.

    Parameters:
        window_size (int): Odd boxcar window side, at least 3. Default ``7``.
        enl (float): Equivalent number of looks. Default ``4.4`` (IW GRDH).
        memory_budget_mb (float): Ceiling for the whole step, in MB; chunk size and
            concurrency are derived from it. Default ``4096``.
        compress (bool): zlib-compress NetCDF output. Default ``False``.
    """

    name = "lee_filter"
    kernel = staticmethod(lee_block)

    def kernel_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        enl = float(params.get("enl", DEFAULT_ENL))
        if not enl > 0:
            raise ValueError(f"enl must be positive, got {enl}.")
        return {"enl": enl}


@StepRegistry.register
class QueganFilterStep(_SpeckleFilterStep):
    """Quegan & Yu multi-temporal speckle filter, out-of-core.

    Accepts a ``.nc`` Tile stack with at least two dates; variables are independent and
    time order is unchanged.  Each date's pixel becomes its local mean times the mean,
    over the valid dates, of value / local mean.  Input must be linear power; invalid
    pixels are excluded and restored as nodata.

    Parameters:
        window_size (int): Odd boxcar window side for the local mean, at least 3. Default ``7``.
        memory_budget_mb (float): Ceiling for the whole step, in MB. Default ``4096``.
        compress (bool): zlib-compress NetCDF output. Default ``False``.
    """

    name = "quegan_filter"
    kernel = staticmethod(quegan_block)
    accepts_tif = False
    min_dates = 2

    def kernel_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}
