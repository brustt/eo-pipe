"""Tests for ClipStep."""

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import box

from eo_pipe.steps.raster.clip import ClipStep


def _make_clip_shp(crs="EPSG:4326"):
    """Small polygon that covers the centre of the 64×64 raster."""
    # Raster bounds: (0, 0, 0.64, 0.64) in degrees
    # Clip to inner quarter: (0.16, 0.16, 0.48, 0.48)
    geom = box(0.16, 0.16, 0.48, 0.48)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)


class TestClipStep:
    def test_output_exists(self, single_raster, output_dir):
        step = ClipStep()
        shp = _make_clip_shp()
        result = step.execute([single_raster], output_dir, shp=shp, save_overlay=False)

        assert len(result.outputs) == 1
        assert result.outputs[0].exists()

    def test_clipped_smaller_than_input(self, single_raster, output_dir):
        step = ClipStep()
        shp = _make_clip_shp()
        result = step.execute([single_raster], output_dir, shp=shp, save_overlay=False)

        with rio.open(single_raster) as src:
            orig_area = src.width * src.height
        with rio.open(result.outputs[0]) as dst:
            clipped_area = dst.width * dst.height

        assert clipped_area < orig_area

    def test_overlay_saved_when_requested(self, single_raster, output_dir):
        step = ClipStep()
        shp = _make_clip_shp()
        result = step.execute([single_raster], output_dir, shp=shp, save_overlay=True)

        key = f"overlay_{single_raster.stem}"
        assert key in result.artifacts
        assert result.artifacts[key].exists()

    def test_overlay_saved_by_default(self, single_raster, output_dir):
        """save_overlay defaults to True — artifact should be present without explicit kwarg."""
        step = ClipStep()
        shp = _make_clip_shp()
        result = step.execute([single_raster], output_dir, shp=shp)
        key = f"overlay_{single_raster.stem}"
        assert key in result.artifacts
        assert result.artifacts[key].exists()

    def test_overlay_not_saved_when_disabled(self, single_raster, output_dir):
        step = ClipStep()
        shp = _make_clip_shp()
        result = step.execute([single_raster], output_dir, shp=shp, save_overlay=False)
        assert not result.artifacts

    def test_multiple_inputs(self, two_rasters, output_dir):
        step = ClipStep()
        shp = _make_clip_shp()
        result = step.execute(two_rasters, output_dir, shp=shp, save_overlay=False)
        assert len(result.outputs) == 2

    def test_crs_mismatch_reprojects(self, single_raster, output_dir):
        """Clip shape in EPSG:3857 should still work — step reprojects it."""
        step = ClipStep()
        shp_3857 = _make_clip_shp(crs="EPSG:4326").to_crs("EPSG:3857")
        result = step.execute([single_raster], output_dir, shp=shp_3857, save_overlay=False)
        assert result.outputs[0].exists()

    def test_step_name(self):
        assert ClipStep.name == "clip"
