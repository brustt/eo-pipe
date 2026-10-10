"""Stack Aligned rasters of each Tile along time into a NetCDF Dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.crs import CRS

from eo_pipe.io.aligned import TAG_PARAMS, identify, iso_to_datetime64
from eo_pipe.io.lattice import Lattice
from eo_pipe.io.output_types import XarrayOutput
from eo_pipe.io.path_utils import NamedPathStrategy
from eo_pipe.logging import setup_logger
from eo_pipe.pipeline.base import StepBase, StepOutput
from eo_pipe.pipeline.registry import StepRegistry

logger = setup_logger("eo_pipe.steps.raster.stack_tiles")


@dataclass
class _Member:
    """One Aligned raster, described by its tags (or filename)."""

    path: Path
    tile: str
    iso: str
    variables: List[str]
    crs: CRS
    res: float
    left: float
    top: float
    width: int
    height: int
    dtype: str
    nodata: Optional[float]
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def right(self) -> float:
        return self.left + self.width * self.res

    @property
    def bottom(self) -> float:
        return self.top - self.height * self.res


def _describe(path: Path) -> _Member:
    with rio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0 or abs(t.a + t.e) > 1e-9:
            raise ValueError(f"{path.name}: expected a north-up raster with square pixels.")
        ident = identify(path.name, dict(src.tags()), src.count, list(src.descriptions))
        return _Member(
            path=path, tile=ident.tile, iso=ident.iso, variables=ident.variables, crs=src.crs,
            res=t.a, left=t.c, top=t.f, width=src.width, height=src.height,
            dtype=src.dtypes[0], nodata=src.nodata, tags=ident.tags,
        )


def _same_nodata(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


def _validate(tile: str, members: List[_Member]) -> None:
    ref = members[0]
    for m in members[1:]:
        if m.crs != ref.crs:
            raise ValueError(f"Tile {tile}: mixed CRS ({ref.path.name} vs {m.path.name}).")
        if not _same_nodata(m.nodata, ref.nodata):
            raise ValueError(
                f"Tile {tile}: mixed nodata ({ref.nodata} in {ref.path.name}, {m.nodata} in {m.path.name})."
            )
        if m.dtype != ref.dtype:
            raise ValueError(
                f"Tile {tile}: mixed dtype ({ref.dtype} in {ref.path.name}, {m.dtype} in {m.path.name})."
            )
        if abs(m.res - ref.res) > 1e-9:
            raise ValueError(
                f"Tile {tile}: mixed resolution ({ref.res} in {ref.path.name}, {m.res} in {m.path.name})."
            )
        if not Lattice(ref.left, ref.top, ref.res).holds(m.left, m.top):
            raise ValueError(
                f"Tile {tile}: {m.path.name} is not on the same lattice as {ref.path.name}."
            )


def _nodata_mask(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    if nodata is None:
        return np.zeros(data.shape, dtype=bool)
    return np.isnan(data) if math.isnan(nodata) else data == nodata


@StepRegistry.register
class StackTilesStep(StepBase):
    """Stack the Aligned rasters of each Tile into one NetCDF Dataset per Tile.

    Expects the outputs of :class:`~eo_pipe.steps.raster.align_tile.AlignToTileStep`
    (use ``MergeBatch``).  Tile, datetime and polarization are read from the GeoTIFF
    tags, falling back to the filename.  Output ``stack_<TILE>.nc`` is an
    ``xarray.Dataset`` with dims ``(time, y, x)`` and one variable per polarization
    (``VV``, ``VH``) or band description (``band_N`` when undescribed).  No
    SAR-specific checks are made: group repeat-pass sets before calling.

    Parameters:
        union (bool): Pad every date to the union of extents instead of cropping
            to their intersection. Default ``False``.
        nodata (float | None): Fill value for padded cells. Default: the first
            input's nodata, else ``0``.
        time_bin (str | None): ``None`` (default) keeps one time step per exact
            datetime. ``"day"`` combines inputs of the same calendar day into one
            step (earliest datetime), e.g. consecutive slices of one pass; where
            they overlap the first valid value wins. Only combine rasters you
            already grouped (same relative orbit and orbit state).
    """

    name = "stack_tiles"

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        union = bool(params.get("union", False))
        time_bin = params.get("time_bin")
        if time_bin not in (None, "day"):
            raise ValueError(f"time_bin must be None or 'day', got {time_bin!r}.")
        groups: Dict[str, List[_Member]] = {}
        for path in inputs:
            member = _describe(Path(path))
            groups.setdefault(member.tile, []).append(member)

        outputs: List[XarrayOutput] = []
        for tile in sorted(groups):
            members = groups[tile]
            _validate(tile, members)
            ds = self._build(tile, members, union, params.get("nodata"), time_bin)
            path = NamedPathStrategy(".nc").resolve(f"stack_{tile}", Path(), 0, output_dir)
            outputs.append(XarrayOutput(ds=ds, path=path))
        return StepOutput(outputs=outputs)

    def _build(
        self,
        tile: str,
        members: List[_Member],
        union: bool,
        nodata_param: Optional[float],
        time_bin: Optional[str],
    ) -> xr.Dataset:
        ref = members[0]
        res = ref.res
        pick = (min, max) if union else (max, min)
        left = pick[0](m.left for m in members)
        right = pick[1](m.right for m in members)
        top = pick[1](m.top for m in members)
        bottom = pick[0](m.bottom for m in members)
        if right - left <= 0 or top - bottom <= 0:
            raise ValueError(f"Tile {tile}: the inputs do not overlap; use union=True to pad them.")
        lattice = Lattice(left, top, res)
        width, height = lattice.shape((left, bottom, right, top))

        nodata = nodata_param if nodata_param is not None else (ref.nodata if ref.nodata is not None else 0)
        bin_of = (lambda iso: iso[:10]) if time_bin == "day" else (lambda iso: iso)
        bins = sorted({bin_of(m.iso) for m in members})
        time_index = {b: i for i, b in enumerate(bins)}
        isos = [min(m.iso for m in members if bin_of(m.iso) == b) for b in bins]
        arrays: Dict[str, np.ndarray] = {}
        seen: set = set()

        for m in members:
            with rio.open(m.path) as src:
                data = src.read()
            valid = ~_nodata_mask(data, m.nodata)
            for band, var in enumerate(m.variables):
                key = (m.iso, var)
                if key in seen:
                    raise ValueError(
                        f"Tile {tile}: duplicate variable '{var}' at {m.iso} ({m.path.name})."
                    )
                seen.add(key)
                if var not in arrays:
                    arrays[var] = np.full((len(isos), height, width), nodata, dtype=ref.dtype)
                col0, row0 = lattice.offset(m.left, m.top)
                c0, c1 = max(0, col0), min(width, col0 + m.width)
                r0, r1 = max(0, row0), min(height, row0 + m.height)
                if c0 >= c1 or r0 >= r1:
                    continue
                block = data[band, r0 - row0 : r1 - row0, c0 - col0 : c1 - col0]
                ok = valid[band, r0 - row0 : r1 - row0, c0 - col0 : c1 - col0]
                target = arrays[var][time_index[bin_of(m.iso)], r0:r1, c0:c1]
                free = ok & _nodata_mask(target, nodata)   # first valid value wins when slices overlap
                target[free] = block[free]

        ds = xr.Dataset(
            {var: (("time", "y", "x"), arr) for var, arr in arrays.items()},
            coords={
                "time": np.array([iso_to_datetime64(i) for i in isos]),
                "y": top - res * (np.arange(height) + 0.5),
                "x": left + res * (np.arange(width) + 0.5),
            },
        )
        ds = ds.rio.write_crs(ref.crs).rio.write_coordinate_system()
        for var in arrays:
            ds[var].encoding["_FillValue"] = nodata
            # Keep spatial_ref a coordinate on plain xr.open_dataset (no decode_coords="all")
            ds[var].encoding.pop("grid_mapping", None)
            ds[var].attrs.pop("grid_mapping", None)
            ds[var].attrs["nodata"] = nodata
        ds.attrs.update(
            tile=tile,
            resolution=res,
            nodata=nodata,
            extent="union" if union else "intersection",
            sources=json.dumps(sorted(m.path.stem for m in members)),
            source_tags=json.dumps({m.path.stem: m.tags for m in members}),
            align_params=json.dumps(sorted({m.tags[TAG_PARAMS] for m in members if TAG_PARAMS in m.tags})),
        )
        return ds
