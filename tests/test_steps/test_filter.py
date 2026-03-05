"""Tests for FilterStep."""

import numpy as np
import rasterio as rio

from eo_pipe.steps.raster.filter import FilterStep


class TestFilterStep:
    def test_output_exists(self, single_raster, output_dir):
        step = FilterStep()
        result = step.execute([single_raster], output_dir, method="median", kernel_size=3)
        assert result.outputs[0].exists()

    def test_output_shape_preserved(self, single_raster, output_dir):
        step = FilterStep()
        result = step.execute([single_raster], output_dir, method="median", kernel_size=3)

        with rio.open(single_raster) as src:
            orig_shape = (src.count, src.height, src.width)
        with rio.open(result.outputs[0]) as dst:
            out_shape = (dst.count, dst.height, dst.width)

        assert orig_shape == out_shape

    def test_median_smooths_data(self, write_raster, output_dir):
        """After median filtering, variance should be lower."""
        rng = np.random.default_rng(11)
        noisy = rng.integers(0, 255, (1, 64, 64), dtype=np.uint8)
        inp = write_raster("noisy.tif", data=noisy, profile={"count": 1})

        step = FilterStep()
        result = step.execute([inp], output_dir, method="median", kernel_size=7)

        with rio.open(result.outputs[0]) as dst:
            filtered = dst.read(1).astype(float)

        assert filtered.std() < noisy[0].astype(float).std()

    def test_multiple_inputs(self, two_rasters, output_dir):
        step = FilterStep()
        result = step.execute(two_rasters, output_dir, method="median", kernel_size=3)
        assert len(result.outputs) == 2

    def test_invalid_method_raises(self, single_raster, output_dir):
        step = FilterStep()
        import pytest
        with pytest.raises(ValueError, match="not supported"):
            step.execute([single_raster], output_dir, method="gaussian")

    def test_default_params(self, single_raster, output_dir):
        step = FilterStep()
        result = step.execute([single_raster], output_dir)
        assert result.outputs[0].exists()

    def test_step_name(self):
        assert FilterStep.name == "filter"
