"""Alignment lattice: a pixel grid anchored at an origin, with whole-pixel arithmetic.

Two rasters on the same lattice differ in origin by a whole number of pixels, so
they combine by array indexing with no resampling.  This module owns every
calculation that relies on that property: snapping extents outward, intersecting
them, checking that a raster sits on the lattice, and converting coordinates to
pixel offsets.  Bounds are ``(minx, miny, maxx, maxy)`` in the lattice CRS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from affine import Affine
    from eo_pipe.io.tiles import Tile

Bounds = Tuple[float, float, float, float]

#: Float noise tolerated, as a fraction of a pixel, when snapping extents.
SNAP_EPS = 1e-6
#: Largest fraction of a pixel by which an origin may miss the lattice.
ALIGN_TOL = 1e-3


def intersect(a: Bounds, b: Bounds) -> Optional[Bounds]:
    """Intersection of two boxes, or ``None`` when they do not overlap."""
    minx, miny = max(a[0], b[0]), max(a[1], b[1])
    maxx, maxy = min(a[2], b[2]), min(a[3], b[3])
    return (minx, miny, maxx, maxy) if minx < maxx and miny < maxy else None


@dataclass(frozen=True)
class Lattice:
    """Pixel grid with upper-left corner ``(origin_x, origin_y)`` and square-or-not pixels."""

    origin_x: float
    origin_y: float
    res_x: float
    res_y: Optional[float] = None

    def __post_init__(self) -> None:
        if self.res_y is None:
            object.__setattr__(self, "res_y", self.res_x)

    @classmethod
    def of_tile(cls, tile: "Tile", res: float) -> "Lattice":
        """The lattice of *tile* at resolution *res*."""
        return cls(tile.ul_x, tile.ul_y, res)

    def snap_outward(self, bounds: Bounds) -> Bounds:
        """Expand *bounds* outward to whole-pixel multiples from the origin."""
        minx, miny, maxx, maxy = bounds
        ox, oy, rx, ry = self.origin_x, self.origin_y, self.res_x, self.res_y
        return (
            ox + math.floor((minx - ox) / rx + SNAP_EPS) * rx,
            oy - math.ceil((oy - miny) / ry - SNAP_EPS) * ry,
            ox + math.ceil((maxx - ox) / rx - SNAP_EPS) * rx,
            oy - math.floor((oy - maxy) / ry + SNAP_EPS) * ry,
        )

    def shape(self, bounds: Bounds) -> Tuple[int, int]:
        """``(width, height)`` in pixels of lattice-snapped *bounds*."""
        minx, miny, maxx, maxy = bounds
        return round((maxx - minx) / self.res_x), round((maxy - miny) / self.res_y)

    def holds(self, x: float, y: float) -> bool:
        """Whether the point ``(x, y)`` is a pixel corner of this lattice."""
        return (
            abs((x - self.origin_x) / self.res_x - round((x - self.origin_x) / self.res_x)) < ALIGN_TOL
            and abs((self.origin_y - y) / self.res_y - round((self.origin_y - y) / self.res_y)) < ALIGN_TOL
        )

    def holds_transform(self, transform: "Affine") -> bool:
        """Whether a north-up raster with this transform sits on the lattice."""
        return (
            transform.b == 0
            and transform.d == 0
            and abs(transform.a - self.res_x) < 1e-9
            and abs(-transform.e - self.res_y) < 1e-9
            and self.holds(transform.c, transform.f)
        )

    def offset(self, x: float, y: float) -> Tuple[int, int]:
        """Whole-pixel ``(col, row)`` of the point ``(x, y)`` from the origin."""
        return round((x - self.origin_x) / self.res_x), round((self.origin_y - y) / self.res_y)

    def anchored_at(self, bounds: Bounds) -> "Lattice":
        """Same lattice with its origin moved to the upper-left of lattice-snapped *bounds*.

        Offsets taken from the result index straight into an array covering *bounds*.
        """
        return Lattice(bounds[0], bounds[3], self.res_x, self.res_y)
