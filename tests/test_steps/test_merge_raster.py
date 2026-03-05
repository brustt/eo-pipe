"""Tests for MergeRasterStep."""

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from eo_pipe.steps.raster.merge import MergeRasterStep


def _make_adjacent_rasters(write_raster):
    """Two 64×64 rasters side-by-side (left/right) sharing the same CRS."""
    crs = CRS.from_epsg(4326)
    rng = np.random.default_rng(5)

    # Left tile: x from 0 to 0.64
    left_data = rng.integers(10, 200, (3, 64, 64), dtype=np.uint8)
    left = write_raster(
        "left.tif",
        data=left_data,
        profile={
            "crs": crs,
            "transform": from_bounds(0, 0, 0.64, 0.64, 64, 64),
        },
    )
    # Right tile: x from 0.64 to 1.28
    right_data = rng.integers(10, 200, (3, 64, 64), dtype=np.uint8)
    right = write_raster(
        "right.tif",
        data=right_data,
        profile={
            "crs": crs,
            "transform": from_bounds(0.64, 0, 1.28, 0.64, 64, 64),
        },
    )
    return [left, right]


class TestMergeRasterStep:
    def test_single_input_passthrough(self, single_raster, output_dir):
        step = MergeRasterStep()
        result = step.execute([single_raster], output_dir)
        assert result.outputs == [single_raster]

    def test_two_rasters_produce_one_output(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        step = MergeRasterStep()
        result = step.execute(inputs, output_dir, to_cog=False)

        assert len(result.outputs) == 1
        assert result.outputs[0].exists()

    def test_merged_wider_than_inputs(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        step = MergeRasterStep()
        result = step.execute(inputs, output_dir, to_cog=False)

        with rio.open(result.outputs[0]) as dst:
            assert dst.width > 64  # merged should be wider than either tile

    def test_output_name_param(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        step = MergeRasterStep()
        result = step.execute(inputs, output_dir, output_name="my_merge", to_cog=False)
        assert result.outputs[0].name == "my_merge.tif"

    def test_step_name(self):
        assert MergeRasterStep.name == "merge_raster"
