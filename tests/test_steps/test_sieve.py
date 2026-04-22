"""Tests for SieveStep (requires system GDAL / osgeo_utils)."""

import numpy as np
import pytest

pytest.importorskip("osgeo.gdal", reason="GDAL C extension not installed")

import rasterio as rio

from eo_pipe.steps.raster.sieve import SieveStep


def _make_classified(write_raster):
    """Single-band uint8 raster with a large region (1) and tiny region (255)."""
    data = np.ones((1, 64, 64), dtype=np.uint8)
    data[0, 0, 0] = 255  # 1-pixel isolated region
    data[0, 1, 0] = 255  # add one more isolated pixel (still < threshold)
    return write_raster(
        "classified.tif",
        data=data,
        profile={"count": 1, "dtype": "uint8", "nodata": 0},
    )


class TestSieveStep:
    def test_output_exists(self, write_raster, output_dir):
        inp = _make_classified(write_raster)
        step = SieveStep()
        result = step.execute([inp], output_dir, threshold=5, connectedness=4).flush_all()
        assert result.outputs[0].exists()

    def test_small_region_removed(self, write_raster, output_dir):
        """Isolated 2-pixel region should be removed with threshold=5."""
        inp = _make_classified(write_raster)
        step = SieveStep()
        result = step.execute([inp], output_dir, threshold=5, connectedness=4).flush_all()

        with rio.open(result.outputs[0]) as dst:
            out_data = dst.read(1)

        assert 255 not in out_data

    def test_large_region_preserved(self, write_raster, output_dir):
        """Main region (value=1, ~4094 pixels) must survive sieving."""
        inp = _make_classified(write_raster)
        step = SieveStep()
        result = step.execute([inp], output_dir, threshold=5, connectedness=4).flush_all()

        with rio.open(result.outputs[0]) as dst:
            out_data = dst.read(1)

        assert 1 in out_data

    def test_default_params(self, write_raster, output_dir):
        inp = _make_classified(write_raster)
        step = SieveStep()
        result = step.execute([inp], output_dir).flush_all()
        assert result.outputs[0].exists()

    def test_step_name(self):
        assert SieveStep.name == "sieve"
