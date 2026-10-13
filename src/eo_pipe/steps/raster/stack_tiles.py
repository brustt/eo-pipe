"""Stack Aligned rasters of each Tile along time into a NetCDF Dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import dask.array as da
import numpy as np
import rasterio as rio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.crs import CRS

from eo_pipe.io.aligned import (
    DEFAULT_DATE_PATTERN,
    TAG_PARAMS,
    datetime_token,
    identify,
    iso_to_datetime64,
    polarization_of,
    token_to_iso,
)
from eo_pipe.io.lattice import Lattice
from eo_pipe.io.memory import budget_mb, plan_chunks
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


def _describe(path: Path, group: Optional[str] = None, date_pattern: str = DEFAULT_DATE_PATTERN) -> _Member:
    """Read a raster's header (no pixels); reject non north-up or non-square grids.

    Without *group* the raster must be an Aligned raster.  With it, the raster is taken
    on its own grid: *group* names the stack, the datetime comes from the filename
    (*date_pattern*) and the variable from its polarization / band descriptions.
    """
    with rio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0 or abs(t.a + t.e) > 1e-9:
            raise ValueError(f"{path.name}: expected a north-up raster with square pixels.")
        if group is None:
            ident = identify(path.name, dict(src.tags()), src.count, list(src.descriptions))
            tile, iso, variables, tags = ident.tile, ident.iso, ident.variables, ident.tags
        else:
            tile = group
            iso = token_to_iso(datetime_token(path.stem, date_pattern))
            pol = polarization_of(src.count, src.descriptions, path.stem)
            variables = [pol] if pol else [d or f"band_{i}" for i, d in enumerate(src.descriptions, start=1)]
            tags = dict(src.tags())
        return _Member(
            path=path, tile=tile, iso=iso, variables=variables, crs=src.crs,
            res=t.a, left=t.c, top=t.f, width=src.width, height=src.height,
            dtype=src.dtypes[0], nodata=src.nodata, tags=tags,
        )


def _same_nodata(a: Optional[float], b: Optional[float]) -> bool:
    """Nodata equality that treats NaN as equal to NaN and ``None`` only as ``None``."""
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


def _validate(tile: str, members: List[_Member]) -> None:
    """Check the members of one Tile share CRS, nodata, dtype, resolution and lattice.

    Header-only, so failures happen before any pixel is read or file written.
    """
    ref = members[0]
    for m in members[1:]:
        if m.crs != ref.crs:
            raise ValueError(f"Stack {tile}: mixed CRS ({ref.path.name} vs {m.path.name}).")
        if not _same_nodata(m.nodata, ref.nodata):
            raise ValueError(
                f"Stack {tile}: mixed nodata ({ref.nodata} in {ref.path.name}, {m.nodata} in {m.path.name})."
            )
        if m.dtype != ref.dtype:
            raise ValueError(
                f"Stack {tile}: mixed dtype ({ref.dtype} in {ref.path.name}, {m.dtype} in {m.path.name})."
            )
        if abs(m.res - ref.res) > 1e-9:
            raise ValueError(
                f"Stack {tile}: mixed resolution ({ref.res} in {ref.path.name}, {m.res} in {m.path.name})."
            )
        if not Lattice(ref.left, ref.top, ref.res).holds(m.left, m.top):
            raise ValueError(
                f"Stack {tile}: {m.path.name} is not on the same lattice as {ref.path.name}."
            )


def _nodata_mask(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Boolean mask of the cells equal to *nodata* (NaN-aware; all ``False`` if ``None``)."""
    if nodata is None:
        return np.zeros(data.shape, dtype=bool)
    return np.isnan(data) if math.isnan(nodata) else data == nodata


def _assemble_block(
    block: np.ndarray,
    block_info: Any = None,
    *,
    entries: List[Tuple[_Member, List[Tuple[int, int, int]]]],
    lattice: Lattice,
    nodata: float,
) -> np.ndarray:
    """Build one ``(variable, time, rows, cols)`` chunk from the source windows it covers.

    *entries* lists, in input order, each source with its ``(band, variable, time step)``
    contributions; a source is opened once and all its needed bands read together.
    """
    (_, _), (_, _), (r_lo, r_hi), (c_lo, c_hi) = block_info[0]["array-location"]
    out = np.full(block.shape, nodata, dtype=block.dtype)
    for m, contributions in entries:     # input order: first valid value wins when slices overlap
        col0, row0 = lattice.offset(m.left, m.top)
        c0, c1 = max(c_lo, col0), min(c_hi, col0 + m.width)
        r0, r1 = max(r_lo, row0), min(r_hi, row0 + m.height)
        if c0 >= c1 or r0 >= r1:
            continue
        window = rio.windows.Window(c0 - col0, r0 - row0, c1 - c0, r1 - r0)
        with rio.open(m.path) as src:
            data = src.read([band + 1 for band, _, _ in contributions], window=window)
        for plane, (_, v, t) in zip(data, contributions):
            target = out[v, t, r0 - r_lo : r1 - r_lo, c0 - c_lo : c1 - c_lo]
            free = ~_nodata_mask(plane, m.nodata) & _nodata_mask(target, nodata)
            target[free] = plane[free]
    return out


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
        group (str | None): Stack rasters that are *not* Aligned rasters, each on its own
            native grid (no ``align_to_tile``).  All inputs form one stack named
            ``stack_<group>.nc``; the datetime is read from the filename (``date_pattern``)
            and the variable from the polarization (``_vv`` / ``_vh`` token) or band
            description.  They must already share one lattice (CRS, resolution, whole-pixel
            offsets), as repeat-pass scenes of one orbit do; otherwise the step raises.  Default ``None``: Aligned
            rasters, grouped by Tile.
        date_pattern (str): With ``group``, regex extracting the acquisition date from the
            filename stem. Default: the *first* ``YYYYMMDD[THHMMSS]`` in the stem, so a
            Sentinel-1 ``<start>_<end>`` name is dated by its start.  Groups of a custom
            pattern are concatenated and must give ``YYYYMMDD[THHMMSS]``.
        memory_budget_mb (float): Ceiling for the whole step, in MB. The Tile stack is
            never fully loaded: it is built lazily in spatial chunks (full time axis per
            chunk) and streamed to the NetCDF file, one Tile after another. Chunk size
            and concurrency are derived from it; very long series get smaller chunks.
            Raises if not even a minimal chunk fits. Default ``4096``.
    """

    name = "stack_tiles"

    def execute(self, inputs: List[Path], output_dir: Path, **params: Any) -> StepOutput:
        """Validate and describe each Tile's stack; pixels are computed at flush time."""
        union = bool(params.get("union", False))
        time_bin = params.get("time_bin")
        if time_bin not in (None, "day"):
            raise ValueError(f"time_bin must be None or 'day', got {time_bin!r}.")
        budget = budget_mb(params)
        group = params.get("group")
        date_pattern = params.get("date_pattern", DEFAULT_DATE_PATTERN)
        groups: Dict[str, List[_Member]] = {}
        for path in inputs:
            member = _describe(Path(path), group, date_pattern)
            groups.setdefault(member.tile, []).append(member)

        outputs: List[XarrayOutput] = []
        for tile in sorted(groups):
            members = groups[tile]
            _validate(tile, members)
            ds, workers = self._build(tile, members, union, params.get("nodata"), time_bin, budget)
            path = NamedPathStrategy(".nc").resolve(f"stack_{tile}", Path(), 0, output_dir)
            outputs.append(XarrayOutput(ds=ds, path=path, num_workers=workers))
        return StepOutput(outputs=outputs)

    def _build(
        self,
        tile: str,
        members: List[_Member],
        union: bool,
        nodata_param: Optional[float],
        time_bin: Optional[str],
        budget: float,
    ) -> Tuple[xr.Dataset, int]:
        """Build the lazy, dask-backed Dataset of one Tile.

        Returns the Dataset and the number of chunks the scheduler may process at once.
        """
        ref = members[0]
        res = ref.res
        # Intersection: tightest bounds; union: widest bounds.
        pick = (min, max) if union else (max, min)
        left = pick[0](m.left for m in members)
        right = pick[1](m.right for m in members)
        top = pick[1](m.top for m in members)
        bottom = pick[0](m.bottom for m in members)
        if right - left <= 0 or top - bottom <= 0:
            raise ValueError(f"Stack {tile}: the inputs do not overlap; use union=True to pad them.")
        lattice = Lattice(left, top, res)
        width, height = lattice.shape((left, bottom, right, top))

        nodata = nodata_param if nodata_param is not None else (ref.nodata if ref.nodata is not None else 0)
        # Time steps: one per datetime, or per calendar day when time_bin="day".
        bin_of = (lambda iso: iso[:10]) if time_bin == "day" else (lambda iso: iso)
        bins = sorted({bin_of(m.iso) for m in members})
        time_index = {b: i for i, b in enumerate(bins)}
        isos = [min(m.iso for m in members if bin_of(m.iso) == b) for b in bins]
        var_index: Dict[str, int] = {}
        entries: List[Tuple[_Member, List[Tuple[int, int, int]]]] = []   # member -> (band, variable, time step)
        seen: set = set()
        for m in members:
            contributions = []
            for band, var in enumerate(m.variables):
                key = (m.iso, var)
                if key in seen:
                    raise ValueError(
                        f"Stack {tile}: duplicate variable '{var}' at {m.iso} ({m.path.name})."
                    )
                seen.add(key)
                contributions.append((band, var_index.setdefault(var, len(var_index)), time_index[bin_of(m.iso)]))
            if contributions:
                entries.append((m, contributions))

        dtype = np.dtype(ref.dtype)
        plan = plan_chunks(budget, len(var_index) * len(isos) * dtype.itemsize, height, width)
        logger.info(
            "Tile %s: %d dates x %d variables, %dx%d px, chunks of %d px, %d at a time",
            tile, len(isos), len(var_index), width, height, plan.side, plan.workers,
        )
        # One chunk holds every variable, so each source is opened once per chunk; the
        # per-variable arrays slice the shared chunk and are computed together at write time.
        base = da.zeros(
            (len(var_index), len(isos), height, width),
            chunks=(len(var_index), len(isos), plan.side, plan.side), dtype=dtype,
        )
        combined = base.map_blocks(_assemble_block, dtype=dtype, entries=entries, lattice=lattice, nodata=nodata)
        arrays: Dict[str, da.Array] = {var: combined[i] for var, i in var_index.items()}

        ds = xr.Dataset(
            {var: (("time", "y", "x"), arr) for var, arr in arrays.items()},
            coords={
                "time": np.array([iso_to_datetime64(i) for i in isos]),
                "y": top - res * (np.arange(height) + 0.5),
                "x": left + res * (np.arange(width) + 0.5),
            },
        )
        # Output contract: CRS as a coordinate, nodata as _FillValue and a "nodata" attr.
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
        return ds, plan.workers
