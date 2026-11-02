"""Tests for MergeRasterStep."""

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from rasterio.merge import merge as rio_merge

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
        result = step.execute([single_raster], output_dir).flush_all()
        assert result.outputs == [single_raster]

    def test_two_rasters_produce_one_output(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        step = MergeRasterStep()
        result = step.execute(inputs, output_dir, to_cog=False).flush_all()

        assert len(result.outputs) == 1
        assert result.outputs[0].exists()

    def test_merged_wider_than_inputs(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        step = MergeRasterStep()
        result = step.execute(inputs, output_dir, to_cog=False).flush_all()

        with rio.open(result.outputs[0]) as dst:
            assert dst.width > 64  # merged should be wider than either tile

    def test_output_name_param(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        step = MergeRasterStep()
        result = step.execute(inputs, output_dir, output_name="my_merge", to_cog=False).flush_all()
        assert result.outputs[0].name == "my_merge.tif"

    def test_step_name(self):
        assert MergeRasterStep.name == "merge_raster"


def _make_row_of_rasters(write_raster, n):
    """*n* 64×64 single-band rasters side-by-side, values distinct per raster."""
    crs = CRS.from_epsg(4326)
    paths = []
    for i in range(n):
        paths.append(write_raster(
            f"r{i}.tif",
            data=np.full((1, 64, 64), 10 + i, dtype=np.uint8),
            profile={
                "count": 1, "crs": crs,
                "transform": from_bounds(0.64 * i, 0, 0.64 * (i + 1), 0.64, 64, 64),
            },
        ))
    return paths


def _reference(paths):
    srcs = [rio.open(p) for p in paths]
    try:
        data, _ = rio_merge(srcs)
    finally:
        for s in srcs:
            s.close()
    return data


class TestMergeRasterLazy:
    def test_matches_rasterio_merge(self, write_raster, output_dir):
        inputs = _make_adjacent_rasters(write_raster)
        out = MergeRasterStep().execute(inputs, output_dir, to_cog=False).flush_all().outputs[0]
        with rio.open(out) as dst:
            np.testing.assert_array_equal(dst.read(), _reference(inputs))

    def test_many_inputs_same_path(self, write_raster, output_dir):
        inputs = _make_row_of_rasters(write_raster, 6)  # used to switch to gdal_merge
        out = MergeRasterStep().execute(inputs, output_dir, to_cog=False).flush_all().outputs[0]
        with rio.open(out) as dst:
            assert dst.width == 64 * 6
            np.testing.assert_array_equal(dst.read(), _reference(inputs))

    def test_small_budget_streams_in_strips_with_identical_result(self, write_raster, output_dir, tmp_path):
        inputs = _make_row_of_rasters(write_raster, 6)
        big = MergeRasterStep().execute(inputs, output_dir, to_cog=False).flush_all().outputs[0]
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        pending = MergeRasterStep().execute(inputs, small_dir, to_cog=False, memory_budget_mb=0.03)
        assert len(pending.outputs[0].windows) > 1
        small = pending.flush_all().outputs[0]
        with rio.open(big) as a, rio.open(small) as b:
            assert a.transform == b.transform
            np.testing.assert_array_equal(a.read(), b.read())

    def test_gap_between_inputs_filled_with_nodata(self, write_raster, output_dir):
        crs = CRS.from_epsg(4326)
        a = write_raster("a.tif", data=np.full((1, 64, 64), 50, np.uint8),
                         profile={"count": 1, "crs": crs, "transform": from_bounds(0, 0, 0.64, 0.64, 64, 64)})
        b = write_raster("b.tif", data=np.full((1, 64, 64), 60, np.uint8),
                         profile={"count": 1, "crs": crs, "transform": from_bounds(1.28, 0, 1.92, 0.64, 64, 64)})
        out = MergeRasterStep().execute([a, b], output_dir, to_cog=False).flush_all().outputs[0]
        with rio.open(out) as dst:
            data = dst.read(1)
            assert dst.width == 192
            assert (data[:, 64:128] == 0).all() and (data[:, :64] == 50).all() and (data[:, 128:] == 60).all()

    def test_does_not_import_gdal(self, write_raster, output_dir, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "osgeo_utils", None)
        monkeypatch.setitem(sys.modules, "osgeo_utils.gdal_merge", None)
        inputs = _make_row_of_rasters(write_raster, 6)
        MergeRasterStep().execute(inputs, output_dir, to_cog=False).flush_all()

    def test_default_output_is_cog(self, write_raster, output_dir):
        inputs = _make_row_of_rasters(write_raster, 2)
        out = MergeRasterStep().execute(inputs, output_dir).flush_all().outputs[0]
        with rio.open(out) as dst:
            assert dst.overviews(1) or dst.width <= 512
            assert dst.profile["tiled"]
            np.testing.assert_array_equal(dst.read(), _reference(inputs))

    def test_only_inputs_touching_a_strip_are_open(self, write_raster, output_dir, monkeypatch):
        from eo_pipe.steps.raster import merge as merge_mod

        crs = CRS.from_epsg(4326)
        inputs = [
            write_raster(
                f"v{i}.tif", data=np.full((1, 64, 64), 10 + i, np.uint8),
                profile={"count": 1, "crs": crs, "transform": from_bounds(0, 0.64 * i, 0.64, 0.64 * (i + 1), 64, 64)},
            )
            for i in range(6)
        ]
        live, peak = set(), [0]
        real_open = rio.open

        class Tracked:
            def __init__(self, ds):
                self.ds = ds

            def __getattr__(self, k):
                return getattr(self.ds, k)

            def __enter__(self):
                self.ds.__enter__()
                live.add(id(self))
                peak[0] = max(peak[0], len(live))
                return self

            def __exit__(self, *a):
                live.discard(id(self))
                return self.ds.__exit__(*a)

        monkeypatch.setattr(merge_mod.rio, "open", lambda p, *a, **k: Tracked(real_open(p, *a, **k)))
        out = MergeRasterStep().execute(inputs, output_dir, to_cog=False, memory_budget_mb=0.02).flush_all().outputs[0]
        # A 109-row strip straddles at most 3 of the 64-row inputs, plus the output being written
        # (rio.open is patched globally); never all 6 at once.
        assert 0 < peak[0] <= 4
        monkeypatch.undo()
        with rio.open(out) as dst:
            np.testing.assert_array_equal(dst.read(), _reference(inputs))
