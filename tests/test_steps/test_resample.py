"""Tests for ResampleStep."""

import rasterio as rio

from eo_pipe.steps.raster.resample import ResampleStep


class TestResampleStep:
    def test_halves_resolution(self, single_raster, output_dir):
        step = ResampleStep()
        result = step.execute([single_raster], output_dir, target_resolution=0.02)

        assert len(result.outputs) == 1
        with rio.open(result.outputs[0]) as dst:
            assert dst.width == 32
            assert dst.height == 32

    def test_output_path_exists(self, single_raster, output_dir):
        step = ResampleStep()
        result = step.execute([single_raster], output_dir, target_resolution=0.02)
        assert result.outputs[0].exists()

    def test_multiple_inputs(self, two_rasters, output_dir):
        step = ResampleStep()
        result = step.execute(two_rasters, output_dir, target_resolution=0.02)
        assert len(result.outputs) == 2
        for p in result.outputs:
            assert p.exists()

    def test_method_nearest(self, single_raster, output_dir):
        step = ResampleStep()
        result = step.execute(
            [single_raster], output_dir,
            target_resolution=0.02, method="nearest"
        )
        assert result.outputs[0].exists()

    def test_nodata_written(self, single_raster, output_dir):
        step = ResampleStep()
        result = step.execute(
            [single_raster], output_dir,
            target_resolution=0.02, nodata_value=255
        )
        with rio.open(result.outputs[0]) as dst:
            assert dst.nodata == 255

    def test_step_name(self):
        assert ResampleStep.name == "resample"
