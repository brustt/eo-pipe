"""Tests for StackTilesStep — Aligned rasters in, one NetCDF Tile stack per Tile out."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from eo_pipe.steps.raster.align_tile import AlignToTileStep
from eo_pipe.steps.raster.stack_tiles import StackTilesStep
from tests.test_steps.test_align_to_tile import make_raster


def align(inputs, out_dir, grid, **params):
    return AlignToTileStep().execute(list(inputs), out_dir, grid_path=grid, **params).flush_all().outputs


def stack(paths, out_dir, **params):
    result = StackTilesStep().execute(list(paths), out_dir, **params).flush_all()
    return result.outputs


@pytest.fixture()
def aligned_dir(tmp_path):
    d = tmp_path / "aligned"
    d.mkdir()
    return d


@pytest.fixture()
def src_dir(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    return d


class TestGroupingAndStructure:
    def test_one_file_per_tile_with_time_and_polarization(self, src_dir, aligned_dir, output_dir, tile_grid):
        srcs = [
            make_raster(src_dir / "s_20250326T051233_vv.tif", 310000.0, 4890000.0),
            make_raster(src_dir / "s_20250314T051233_vv.tif", 310000.0, 4890000.0),
            make_raster(src_dir / "s_20250314T051233_vh.tif", 310000.0, 4890000.0),
        ]
        outs = stack(align(srcs, aligned_dir, tile_grid), output_dir)
        assert [p.name for p in outs] == ["stack_31TCJ.nc"]
        ds = xr.open_dataset(outs[0])
        assert set(ds.data_vars) == {"VV", "VH"}
        assert ds.sizes["time"] == 2 and ds.sizes["x"] == 64 and ds.sizes["y"] == 64

    def test_time_axis_is_chronological_whatever_the_input_order(self, src_dir, aligned_dir, output_dir, tile_grid):
        late = make_raster(src_dir / "s_20250326_vv.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 7, "float32"))
        early = make_raster(src_dir / "s_20250314_vv.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 3, "float32"))
        out = stack(align([late, early], aligned_dir, tile_grid), output_dir)[0]
        ds = xr.open_dataset(out)
        assert list(ds.time.values) == [np.datetime64("2025-03-14"), np.datetime64("2025-03-26")]
        assert float(ds.VV.isel(time=0).mean()) == 3.0
        assert float(ds.VV.isel(time=1).mean()) == 7.0

    def test_scene_in_two_tiles_gives_two_stacks(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "a_20250314_vv.tif", 409400.0, 4890000.0)
        b = make_raster(src_dir / "b_20250326_vv.tif", 409400.0, 4890000.0)
        outs = stack(align([a, b], aligned_dir, tile_grid), output_dir)
        assert sorted(p.name for p in outs) == ["stack_31TCJ.nc", "stack_31TDJ.nc"]

    def test_crs_is_readable_with_rioxarray(self, src_dir, aligned_dir, output_dir, tile_grid):
        s = make_raster(src_dir / "a_20250314_vv.tif", 310000.0, 4890000.0)
        ds = xr.open_dataset(stack(align([s], aligned_dir, tile_grid), output_dir)[0])
        assert ds.rio.crs.to_epsg() == 32631
        assert ds.x.values[0] == 310005.0  # pixel centre of the first column


class TestExtent:
    def _two_dates(self, src_dir, aligned_dir, tile_grid):
        a = make_raster(src_dir / "a_20250314_vv.tif", 310000.0, 4890000.0)
        b = make_raster(src_dir / "b_20250326_vv.tif", 310100.0, 4890000.0)  # shifted 10 px east
        return align([a, b], aligned_dir, tile_grid)

    def test_intersection_by_default(self, src_dir, aligned_dir, output_dir, tile_grid):
        ds = xr.open_dataset(stack(self._two_dates(src_dir, aligned_dir, tile_grid), output_dir)[0])
        assert ds.sizes["x"] == 54
        assert not ds.VV.isnull().any()

    def test_union_pads_with_nodata(self, src_dir, aligned_dir, output_dir, tile_grid):
        ds = xr.open_dataset(stack(self._two_dates(src_dir, aligned_dir, tile_grid), output_dir, union=True)[0])
        assert ds.sizes["x"] == 74
        assert bool(ds.VV.isel(time=0, x=slice(64, None)).isnull().all())
        assert bool(ds.VV.isel(time=1, x=slice(0, 10)).isnull().all())


class TestMetadata:
    def test_dataset_keeps_tile_resolution_sources_and_tags(self, src_dir, aligned_dir, output_dir, tile_grid):
        s = make_raster(src_dir / "a_20250314_vv.tif", 310000.0, 4890000.0, tags={"orbit": "ascending"})
        ds = xr.open_dataset(stack(align([s], aligned_dir, tile_grid), output_dir)[0])
        assert ds.attrs["tile"] == "31TCJ"
        assert ds.attrs["resolution"] == 10.0
        assert "a_20250314_vv" in ds.attrs["sources"]
        assert "ascending" in ds.attrs["source_tags"]
        assert "resolution" in ds.attrs["align_params"]

    def test_multiband_input_uses_band_descriptions_as_variables(self, src_dir, aligned_dir, output_dir, tile_grid):
        s = make_raster(src_dir / "a_20250314.tif", 310000.0, 4890000.0, count=2, descriptions=["VV", "VH"])
        ds = xr.open_dataset(stack(align([s], aligned_dir, tile_grid), output_dir)[0])
        assert set(ds.data_vars) == {"VV", "VH"}

    def test_undescribed_bands_are_named_band_n(self, src_dir, aligned_dir, output_dir, tile_grid):
        s = make_raster(src_dir / "a_20250314.tif", 310000.0, 4890000.0, count=2)
        ds = xr.open_dataset(stack(align([s], aligned_dir, tile_grid), output_dir)[0])
        assert set(ds.data_vars) == {"band_1", "band_2"}


class TestFilenameFallback:
    def test_untagged_files_are_grouped_by_filename(self, aligned_dir, output_dir):
        for token in ("20250314T051233", "20250326T051233"):
            make_raster(aligned_dir / f"align_31TCJ_{token}_s_{token}_vv.tif", 310000.0, 4890000.0)
        outs = stack(sorted(aligned_dir.glob("*.tif")), output_dir)
        ds = xr.open_dataset(outs[0])
        assert ds.sizes["time"] == 2
        assert "VV" in ds.data_vars   # polarization from the stem suffix

    def test_no_tags_and_no_contract_name_raises_naming_the_file(self, aligned_dir, output_dir):
        bad = make_raster(aligned_dir / "mystery.tif", 310000.0, 4890000.0)
        with pytest.raises(ValueError, match="mystery.tif"):
            stack([bad], output_dir)


class TestGuards:
    def test_mixed_resolution_raises(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "a_20250314_vv.tif", 310000.0, 4890000.0)
        b = make_raster(src_dir / "b_20250326_vv.tif", 310000.0, 4890000.0)
        outs = align([a], aligned_dir, tile_grid) + align([b], aligned_dir, tile_grid, resolution=20.0)
        with pytest.raises(ValueError, match="resolution"):
            stack(outs, output_dir)

    def test_duplicate_date_and_variable_raises(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "a_20250314_vv.tif", 310000.0, 4890000.0)
        b = make_raster(src_dir / "b_20250314_vv.tif", 310000.0, 4890000.0)
        with pytest.raises(ValueError, match="duplicate"):
            stack(align([a, b], aligned_dir, tile_grid), output_dir)

    def test_off_lattice_inputs_raise(self, aligned_dir, output_dir):
        a = make_raster(aligned_dir / "align_31TCJ_20250314_s_vv.tif", 310000.0, 4890000.0)
        b = make_raster(aligned_dir / "align_31TCJ_20250326_s_vv.tif", 310003.0, 4890000.0)
        with pytest.raises(ValueError, match="lattice"):
            stack([a, b], output_dir)

    def test_disjoint_intersection_raises(self, aligned_dir, output_dir):
        a = make_raster(aligned_dir / "align_31TCJ_20250314_s_vv.tif", 310000.0, 4890000.0)
        b = make_raster(aligned_dir / "align_31TCJ_20250326_s_vv.tif", 320000.0, 4890000.0)
        with pytest.raises(ValueError, match="overlap"):
            stack([a, b], output_dir)


class TestRegistration:
    def test_registered_by_name(self):
        from eo_pipe.pipeline.registry import StepRegistry

        assert StackTilesStep.name == "stack_tiles"
        assert "stack_tiles" in StepRegistry.available()


class TestPipeline:
    def test_align_then_stack_in_a_composition(self, src_dir, tmp_path, tile_grid):
        from eo_pipe import MergeBatch, ParallelBatch, PipelineComposition

        srcs = [
            make_raster(src_dir / "s_20250314T051233_vv.tif", 310000.0, 4890000.0),
            make_raster(src_dir / "s_20250326T051233_vv.tif", 310037.0, 4890011.0),
        ]
        ctx = (
            PipelineComposition(workspace=tmp_path / "run")
            .add_step("align_to_tile", ParallelBatch(), grid_path=tile_grid)
            .add_step("stack_tiles", MergeBatch())
            .run(inputs=srcs, save_intermediate=True)
        )
        assert [p.name for p in ctx.inputs] == ["stack_31TCJ.nc"]
        ds = xr.open_dataset(ctx.inputs[0])
        assert ds.sizes["time"] == 2


class TestMoreGuards:
    def test_mixed_crs_raises(self, aligned_dir, output_dir):
        a = make_raster(aligned_dir / "align_31TCJ_20250314_s_vv.tif", 310000.0, 4890000.0)
        b = make_raster(aligned_dir / "align_31TCJ_20250326_s_vv.tif", 310000.0, 4890000.0, crs="EPSG:32632")
        with pytest.raises(ValueError, match="CRS"):
            stack([a, b], output_dir)

    def test_mixed_dtype_raises(self, aligned_dir, output_dir):
        a = make_raster(aligned_dir / "align_31TCJ_20250314_s_vv.tif", 310000.0, 4890000.0)
        b = make_raster(
            aligned_dir / "align_31TCJ_20250326_s_vv.tif", 310000.0, 4890000.0,
            dtype="uint8", fill=np.full((1, 64, 64), 3, "uint8"),
        )
        with pytest.raises(ValueError, match="dtype"):
            stack([a, b], output_dir)

    def test_mixed_nodata_raises(self, aligned_dir, output_dir):
        a = make_raster(aligned_dir / "align_31TCJ_20250314_s_vv.tif", 310000.0, 4890000.0)
        b = make_raster(aligned_dir / "align_31TCJ_20250326_s_vv.tif", 310000.0, 4890000.0, nodata=-1.0)
        with pytest.raises(ValueError, match="nodata"):
            stack([a, b], output_dir)


class TestXarrayOutput:
    def test_flush_writes_a_netcdf_file(self, tmp_path):
        from eo_pipe.io.output_types import XarrayOutput

        ds = xr.Dataset({"VV": (("y", "x"), np.ones((2, 3), "float32"))})
        path = XarrayOutput(ds=ds, path=tmp_path / "sub" / "out.nc").flush()
        assert path.exists()
        assert xr.open_dataset(path).VV.shape == (2, 3)


class TestTimeBinning:
    def _slices(self, src_dir):
        left = np.zeros((1, 64, 64), "float32"); left[0, :, :32] = 4.0
        right = np.zeros((1, 64, 64), "float32"); right[0, :, 32:] = 6.0
        return [
            make_raster(src_dir / "s_20250314T051208_vv.tif", 310000.0, 4890000.0, fill=left),
            make_raster(src_dir / "s_20250314T051233_vv.tif", 310000.0, 4890000.0, fill=right),
        ]

    def test_slices_of_one_day_stay_separate_by_default(self, src_dir, aligned_dir, output_dir, tile_grid):
        ds = xr.open_dataset(stack(align(self._slices(src_dir), aligned_dir, tile_grid), output_dir)[0])
        assert ds.sizes["time"] == 2

    def test_day_binning_combines_slices_into_one_time_step(self, src_dir, aligned_dir, output_dir, tile_grid):
        outs = stack(align(self._slices(src_dir), aligned_dir, tile_grid), output_dir, time_bin="day")
        ds = xr.open_dataset(outs[0])
        assert ds.sizes["time"] == 1
        assert ds.time.values[0] == np.datetime64("2025-03-14T05:12:08")   # earliest slice
        row = ds.VV.isel(time=0, y=10).values
        assert (row[:32] == 4.0).all() and (row[32:64] == 6.0).all()

    def test_overlapping_slices_keep_the_first_valid_value(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "s_20250314T051208_vv.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 4.0, "float32"))
        b = make_raster(src_dir / "s_20250314T051233_vv.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 6.0, "float32"))
        ds = xr.open_dataset(stack(align([a, b], aligned_dir, tile_grid), output_dir, time_bin="day")[0])
        assert float(ds.VV.mean()) == 4.0

    def test_different_days_remain_different_time_steps(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "s_20250314T051208_vv.tif", 310000.0, 4890000.0)
        b = make_raster(src_dir / "s_20250326T051208_vv.tif", 310000.0, 4890000.0)
        ds = xr.open_dataset(stack(align([a, b], aligned_dir, tile_grid), output_dir, time_bin="day")[0])
        assert ds.sizes["time"] == 2

    def test_unknown_time_bin_raises(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "s_20250314T051208_vv.tif", 310000.0, 4890000.0)
        with pytest.raises(ValueError, match="time_bin"):
            stack(align([a], aligned_dir, tile_grid), output_dir, time_bin="week")


def _assert_same_stack(a: Path, b: Path) -> None:
    with xr.open_dataset(a) as da, xr.open_dataset(b) as db:
        xr.testing.assert_identical(da, db)


class TestMemoryBudget:
    """A tiny Memory budget forces many chunks; content must not depend on it."""

    TINY = 0.05  # MB: a handful of pixel rows/columns per chunk

    @pytest.fixture()
    def series(self, src_dir, aligned_dir, tile_grid):
        srcs = []
        for day, x0 in ((12, 310000.0), (14, 310200.0), (16, 310100.0)):
            for pol in ("vv", "vh"):
                srcs.append(make_raster(src_dir / f"s_202503{day}T051233_{pol}.tif", x0, 4890000.0))
        return align(srcs, aligned_dir, tile_grid)

    def test_tiny_budget_equals_large_budget_on_intersection(self, series, tmp_path):
        big = stack(series, tmp_path / "big", memory_budget_mb=4096)[0]
        small = stack(series, tmp_path / "small", memory_budget_mb=self.TINY)[0]
        _assert_same_stack(big, small)

    def test_tiny_budget_equals_large_budget_on_union(self, series, tmp_path):
        big = stack(series, tmp_path / "big", union=True)[0]
        small = stack(series, tmp_path / "small", union=True, memory_budget_mb=self.TINY)[0]
        _assert_same_stack(big, small)
        with xr.open_dataset(small) as ds:
            assert ds.sizes["x"] > 64

    def test_tiny_budget_equals_large_budget_with_overlapping_day_slices(self, src_dir, aligned_dir, tile_grid, tmp_path):
        srcs = [
            make_raster(src_dir / "s_20250314T051233_vv.tif", 310000.0, 4890000.0),
            make_raster(src_dir / "s_20250314T051300_vv.tif", 310200.0, 4890000.0),
        ]
        series = align(srcs, aligned_dir, tile_grid)
        big = stack(series, tmp_path / "big", union=True, time_bin="day")[0]
        small = stack(series, tmp_path / "small", union=True, time_bin="day", memory_budget_mb=self.TINY)[0]
        _assert_same_stack(big, small)
        with xr.open_dataset(small) as ds:
            assert ds.sizes["time"] == 1

    def test_default_budget_is_accepted_and_content_is_unchanged(self, series, tmp_path):
        out = stack(series, tmp_path / "o")[0]
        with xr.open_dataset(out) as ds:
            assert ds.VV.dims == ("time", "y", "x") and ds.attrs["nodata"] == 0.0


class TestOutOfCore:
    def _series(self, src_dir, aligned_dir, tile_grid, n=12, size=256):
        srcs = [
            make_raster(
                src_dir / f"s_{date(2025, 1, 1) + timedelta(days=d):%Y%m%d}T051233_vv.tif", 310000.0, 4890000.0, size=size,
                fill=np.full((1, size, size), d + 1, "float32"),
            )
            for d in range(n)
        ]
        return align(srcs, aligned_dir, tile_grid)

    def test_budget_too_small_for_a_minimal_chunk_raises_naming_the_minimum(self, src_dir, aligned_dir, output_dir, tile_grid):
        series = self._series(src_dir, aligned_dir, tile_grid, n=3, size=64)
        with pytest.raises(ValueError, match=r"memory_budget_mb.*at least"):
            StackTilesStep().execute(series, output_dir, memory_budget_mb=1e-6)

    def test_peak_allocation_stays_well_under_the_stack_size(self, src_dir, aligned_dir, output_dir, tile_grid):
        import tracemalloc

        series = self._series(src_dir, aligned_dir, tile_grid, n=40, size=512)  # stack: 40 MiB
        pending = StackTilesStep().execute(series, output_dir, memory_budget_mb=4)
        tracemalloc.start()
        try:
            pending.flush_all()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 20 * 2**20
        with xr.open_dataset(output_dir / "stack_31TCJ.nc") as ds:
            assert ds.sizes["time"] == 40

    def test_stack_is_chunked_on_disk_with_the_full_time_axis(self, src_dir, aligned_dir, output_dir, tile_grid):
        series = self._series(src_dir, aligned_dir, tile_grid, n=4, size=256)
        out = stack(series, output_dir, memory_budget_mb=0.5)[0]
        with xr.open_dataset(out) as ds:
            chunks = ds.VV.encoding["chunksizes"]
        assert chunks[0] == 4 and chunks[1] < 256

    def test_validation_error_leaves_no_file(self, src_dir, aligned_dir, output_dir, tile_grid):
        a = make_raster(src_dir / "s_20250314_vv.tif", 310000.0, 4890000.0)
        b = make_raster(src_dir / "s_20250326_vv.tif", 310000.0, 4890000.0, dtype="float64", fill=np.ones((1, 64, 64)))
        series = align([a, b], aligned_dir, tile_grid)
        with pytest.raises(ValueError, match="dtype"):
            stack(series, output_dir)
        assert not list(output_dir.glob("*"))

    def test_failed_write_leaves_no_valid_looking_stack(self, src_dir, aligned_dir, output_dir, tile_grid):
        series = self._series(src_dir, aligned_dir, tile_grid, n=3, size=64)
        pending = StackTilesStep().execute(series, output_dir, memory_budget_mb=0.05)
        series[1].write_bytes(b"not a tiff")  # source corrupted between execute and flush
        with pytest.raises(Exception):
            pending.flush_all()
        assert not (output_dir / "stack_31TCJ.nc").exists()


class TestMultiPolarizationSource:
    """A source carrying VV and VH together stacks the same whatever the budget."""

    def test_tiny_budget_equals_large_budget_and_keeps_each_band(self, src_dir, aligned_dir, tile_grid, tmp_path):
        rng = np.random.default_rng(5)
        srcs = []
        for day, x0, vv, vh in ((12, 310000.0, 3.0, 30.0), (14, 310200.0, 7.0, 70.0), (16, 310100.0, 5.0, 50.0)):
            data = np.stack([np.full((64, 64), vv), np.full((64, 64), vh)]).astype("float32")
            data += rng.uniform(0, 0.5, data.shape).astype("float32")
            srcs.append(make_raster(
                src_dir / f"s_202503{day}T051233.tif", x0, 4890000.0, count=2,
                descriptions=("VV", "VH"), fill=data,
            ))
        series = align(srcs, aligned_dir, tile_grid)
        big = stack(series, tmp_path / "big", union=True)[0]
        small = stack(series, tmp_path / "small", union=True, memory_budget_mb=0.05)[0]
        _assert_same_stack(big, small)
        with xr.open_dataset(small) as ds:
            assert set(ds.data_vars) == {"VV", "VH"}
            assert float(ds.VH.isel(time=1).max()) > 69.0 > float(ds.VV.isel(time=1).max()) 


class TestFailedWrite:
    def test_failed_write_leaves_neither_the_file_nor_a_part_file(self, src_dir, aligned_dir, output_dir, tile_grid):
        srcs = [make_raster(src_dir / f"s_2025031{d}T051233_vv.tif", 310000.0, 4890000.0) for d in (2, 4)]
        series = align(srcs, aligned_dir, tile_grid)
        pending = StackTilesStep().execute(series, output_dir, memory_budget_mb=0.05)
        series[1].unlink()                              # a source vanishes before the write
        with pytest.raises(Exception):
            pending.flush_all()
        assert not list(output_dir.glob("*"))


class TestNativeGroup:
    """``group``: raw rasters on a shared native grid are stacked without align_to_tile."""

    def test_stacks_untagged_rasters_on_their_own_grid(self, src_dir, output_dir):
        early = make_raster(src_dir / "S1A_20250314T051233_x_vv.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 3, "float32"))
        late = make_raster(src_dir / "S1A_20250326T051233_x_vv.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 7, "float32"))
        vh = make_raster(src_dir / "S1A_20250314T051233_x_vh.tif", 310000.0, 4890000.0, fill=np.full((1, 64, 64), 5, "float32"))
        outs = stack([late, vh, early], output_dir, group="ro008_descending")
        assert [p.name for p in outs] == ["stack_ro008_descending.nc"]
        ds = xr.open_dataset(outs[0])
        assert set(ds.data_vars) == {"VV", "VH"}
        assert list(ds.time.values) == [np.datetime64("2025-03-14T05:12:33"), np.datetime64("2025-03-26T05:12:33")]
        assert float(ds.VV.isel(time=0).mean()) == 3.0 and float(ds.VV.isel(time=1).mean()) == 7.0
        assert float(ds.VH.isel(time=0).mean()) == 5.0
        assert float(ds.x[0]) == 310005.0 and ds.attrs["tile"] == "ro008_descending"

    def test_without_group_untagged_rasters_are_refused(self, src_dir, output_dir):
        raw = make_raster(src_dir / "S1A_20250314T051233_x_vv.tif", 310000.0, 4890000.0)
        with pytest.raises(ValueError, match="TILE/DATETIME"):
            StackTilesStep().execute([raw], output_dir)
