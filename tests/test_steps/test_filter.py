"""Tests for FilterStep."""

import numpy as np
import rasterio as rio

from eo_pipe.steps.raster.filter import FilterStep, get_filter_method


class TestFilterStep:
    def test_output_exists(self, single_raster, output_dir):
        step = FilterStep()
        result = step.execute([single_raster], output_dir, method="median", kernel_size=3).flush_all()
        assert result.outputs[0].exists()

    def test_output_shape_preserved(self, single_raster, output_dir):
        step = FilterStep()
        result = step.execute([single_raster], output_dir, method="median", kernel_size=3).flush_all()

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
        result = step.execute([inp], output_dir, method="median", kernel_size=7).flush_all()

        with rio.open(result.outputs[0]) as dst:
            filtered = dst.read(1).astype(float)

        assert filtered.std() < noisy[0].astype(float).std()

    def test_multiple_inputs(self, two_rasters, output_dir):
        step = FilterStep()
        result = step.execute(two_rasters, output_dir, method="median", kernel_size=3).flush_all()
        assert len(result.outputs) == 2

    def test_invalid_method_raises(self, single_raster, output_dir):
        step = FilterStep()
        import pytest
        with pytest.raises(ValueError, match="not supported"):
            step.execute([single_raster], output_dir, method="gaussian")

    def test_default_params(self, single_raster, output_dir):
        step = FilterStep()
        result = step.execute([single_raster], output_dir).flush_all()
        assert result.outputs[0].exists()

    def test_nodata_pixels_not_bleed(self, write_raster, output_dir):
        """Nodata pixels must be restored after filtering, not contaminated by neighbours."""
        rng = np.random.default_rng(42)
        data = rng.integers(50, 200, (3, 64, 64), dtype=np.uint8)
        data[:, 0, :] = 0  # top row is nodata (value 0, matches conftest nodata)

        inp = write_raster("nodata_src.tif", data=data)
        step = FilterStep()
        result = step.execute([inp], output_dir, method="median", kernel_size=7).flush_all()

        with rio.open(result.outputs[0]) as dst:
            out = dst.read()

        # Nodata row must still be 0, not contaminated by filter kernel
        np.testing.assert_array_equal(out[:, 0, :], 0)

    def test_step_name(self):
        assert FilterStep.name == "filter"


class TestOtsu:
    def test_binarises_bimodal_band(self, write_raster, output_dir):
        data = np.full((1, 64, 64), 20, dtype=np.uint8)
        data[:, :, 32:] = 200
        inp = write_raster("bimodal.tif", data=data, profile={"count": 1})

        result = FilterStep().execute([inp], output_dir, method="otsu").flush_all()

        with rio.open(result.outputs[0]) as dst:
            out = dst.read(1)
        assert set(np.unique(out)) == {0, 1}
        np.testing.assert_array_equal(out[:, :32], 0)
        np.testing.assert_array_equal(out[:, 32:], 1)

    def test_nodata_excluded_and_restored(self, write_raster, output_dir):
        data = np.full((1, 64, 64), 20, dtype=np.uint8)
        data[:, :, 32:] = 200
        data[:, 0, :] = 0  # nodata row
        inp = write_raster("nodata_otsu.tif", data=data, profile={"count": 1})

        result = FilterStep().execute([inp], output_dir, method="otsu").flush_all()

        with rio.open(result.outputs[0]) as dst:
            out = dst.read(1)
        np.testing.assert_array_equal(out[0, :], 0)
        np.testing.assert_array_equal(out[1:, 32:], 1)

    def test_constant_band_is_zero(self, write_raster, output_dir):
        data = np.full((1, 64, 64), 100, dtype=np.uint8)
        inp = write_raster("const.tif", data=data, profile={"count": 1})
        result = FilterStep().execute([inp], output_dir, method="otsu").flush_all()
        with rio.open(result.outputs[0]) as dst:
            assert dst.read(1).max() == 0


class TestFactory:
    def test_returns_callable(self):
        assert callable(get_filter_method("median"))
        assert callable(get_filter_method("otsu"))

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="not supported"):
            get_filter_method("nope")
