"""Shared pytest fixtures for eo-pipe tests.

All raster fixtures produce real GeoTIFF files written to ``tmp_path``.
No rasterio operations are mocked — tests exercise the actual I/O.

Default raster spec:
    - Size:   64 × 64 px
    - CRS:    EPSG:4326
    - Bounds: (0, 0, 0.64, 0.64)  → 0.01 deg/px
    - dtype:  uint8
    - nodata: 0
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

W, H = 64, 64
CRS_4326 = CRS.from_epsg(4326)
TRANSFORM = from_bounds(0, 0, 0.64, 0.64, W, H)

DEFAULT_PROFILE = {
    "driver": "GTiff",
    "dtype": "uint8",
    "width": W,
    "height": H,
    "count": 3,
    "crs": CRS_4326,
    "transform": TRANSFORM,
    "nodata": 0,
}


# ---------------------------------------------------------------------------
# Low-level writer helper (not a fixture — plain function)
# ---------------------------------------------------------------------------


def _write_raster(
    path: Path,
    data: Optional[np.ndarray] = None,
    profile: Optional[Dict] = None,
) -> Path:
    """Write a synthetic GeoTIFF.  Returns the path."""
    p = dict(DEFAULT_PROFILE)
    if profile:
        p.update(profile)

    if data is None:
        rng = np.random.default_rng(42)
        data = rng.integers(10, 240, size=(p["count"], H, W), dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(path, "w", **p) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def write_raster(tmp_path: Path) -> Callable:
    """Factory: ``write_raster(name, data=None, profile=None) -> Path``."""

    def _factory(
        name: str,
        data: Optional[np.ndarray] = None,
        profile: Optional[Dict] = None,
    ) -> Path:
        return _write_raster(tmp_path / name, data, profile)

    return _factory


@pytest.fixture()
def single_raster(write_raster) -> Path:
    """One 64×64 uint8 3-band raster."""
    return write_raster("input.tif")


@pytest.fixture()
def two_rasters(write_raster) -> list:
    """Two 64×64 uint8 3-band rasters with different random data."""
    rng = np.random.default_rng(7)
    a = rng.integers(10, 200, (3, H, W), dtype=np.uint8)
    b = rng.integers(50, 240, (3, H, W), dtype=np.uint8)
    return [
        write_raster("a.tif", data=a),
        write_raster("b.tif", data=b),
    ]


@pytest.fixture()
def single_band_raster(write_raster) -> Path:
    """Single-band float32 raster (typical for damage ratio output)."""
    rng = np.random.default_rng(13)
    data = rng.random((1, H, W)).astype(np.float32)
    return write_raster(
        "single.tif",
        data=data,
        profile={"count": 1, "dtype": "float32", "nodata": -9999.0},
    )


@pytest.fixture()
def nodata_raster(write_raster) -> Path:
    """Raster with a nodata border (top row = 0 = nodata)."""
    rng = np.random.default_rng(99)
    data = rng.integers(10, 240, (3, H, W), dtype=np.uint8)
    data[:, 0, :] = 0   # nodata row
    return write_raster("nodata.tif", data=data)


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out
