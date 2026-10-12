"""Tests for LeeFilterStep and QueganFilterStep — driven through execute() and flush."""

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_origin

from eo_pipe.io.memory import budget_mb
from eo_pipe.io.speckle import plan_filter_chunks
from eo_pipe.steps.raster.speckle import LeeFilterStep, QueganFilterStep

H = W = 40
NODATA = -9999.0


def reference_means(x, valid, win):
    """Brute-force normalised boxcar mean with an image-edge-truncated window."""
    h = win // 2
    mean = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = valid[max(0, i - h): i + h + 1, max(0, j - h): j + h + 1]
            s = x[max(0, i - h): i + h + 1, max(0, j - h): j + h + 1]
            mean[i, j] = s[v].mean() if v.any() else 0.0
    return mean


def reference_lee(x, valid, win, enl):
    m = reference_means(np.where(valid, x, 0.0), valid, win)
    m2 = reference_means(np.where(valid, x * x, 0.0), valid, win)
    var = np.maximum(m2 - m**2, 0)
    ci2 = np.divide(var, m**2, out=np.zeros_like(var), where=m != 0)
    cu2 = 1 / enl
    w = np.where(ci2 > cu2, (ci2 - cu2) / np.where(ci2 > 0, ci2, 1), 0.0)
    return m + w * (x - m)


def reference_quegan(cube, valid, win):
    means = np.stack([reference_means(np.where(v, c, 0.0), v, win) for c, v in zip(cube, valid)])
    usable = valid & (means > 0)
    ratio = np.where(usable, np.divide(cube, means, out=np.zeros_like(cube), where=usable), 0.0)
    n = usable.sum(0)
    mr = np.where(n > 0, ratio.sum(0) / np.maximum(n, 1), 1.0)
    return means * mr


def speckled(shape, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.gamma(4.4, 1 / 4.4, shape) * 0.1).astype("float32")


def write_tif(path, data, nodata=NODATA, units=None):
    with rio.open(
        path, "w", driver="GTiff", dtype="float32", count=data.shape[0], height=H, width=W,
        crs=CRS.from_epsg(32631), transform=from_origin(300000, 4900000, 10, 10), nodata=nodata,
    ) as dst:
        dst.write(data)
        dst.update_tags(TILE="31TCJ", KEEP="me")
        if units:
            dst.update_tags(units=units)
    return path


def write_stack(path, cubes, units=None, nodata=NODATA):
    """cubes: {var: (time, H, W)}"""
    t = next(iter(cubes.values())).shape[0]
    ds = xr.Dataset(
        {v: (("time", "y", "x"), np.where(np.isnan(c), nodata, c).astype("float32")) for v, c in cubes.items()},
        coords={
            "time": np.array([f"2025-03-{10 + 2 * i}" for i in range(t)], dtype="datetime64[ns]"),
            "y": 4900000 - 10 * (np.arange(H) + 0.5),
            "x": 300000 + 10 * (np.arange(W) + 0.5),
        },
        attrs={"tile": "31TCJ"},
    )
    for v in cubes:
        ds[v].attrs["nodata"] = nodata
        ds[v].encoding["_FillValue"] = nodata
        if units:
            ds[v].attrs["units"] = units
    ds.to_netcdf(path)
    return path


def run(step, inp, out_dir, **params):
    return step.execute([inp], out_dir, **params).flush_all().outputs[0]


# -- Lee ------------------------------------------------------------------------------------


class TestLee:
    def test_tif_matches_reference_and_preserves_metadata(self, tmp_path, output_dir):
        data = speckled((2, H, W))
        inp = write_tif(tmp_path / "a.tif", data)
        out = run(LeeFilterStep(), inp, output_dir, window_size=5, enl=4.4)
        assert out.name == "lee_filter_a.tif"
        with rio.open(out) as src, rio.open(inp) as orig:
            got = src.read()
            assert src.crs == orig.crs and src.transform == orig.transform and src.nodata == NODATA
            assert src.tags()["KEEP"] == "me"
        for b in range(2):
            expected = reference_lee(data[b].astype("float64"), np.ones((H, W), bool), 5, 4.4)
            np.testing.assert_allclose(got[b], expected, rtol=1e-4, atol=1e-6)

    def test_multi_chunk_equals_single_chunk(self, tmp_path, output_dir):
        inp = write_tif(tmp_path / "a.tif", speckled((3, H, W)))
        big = run(LeeFilterStep(), inp, output_dir, window_size=7)
        (output_dir / big.name).rename(output_dir / "big.tif")
        # a tiny budget forces the minimum 16 px chunks
        small_dir = output_dir / "small"
        small_dir.mkdir()
        small = run(LeeFilterStep(), inp, small_dir, window_size=7, memory_budget_mb=0.2)
        with rio.open(output_dir / "big.tif") as a, rio.open(small) as b:
            np.testing.assert_allclose(a.read(), b.read(), rtol=1e-6)

    def test_nodata_excluded_and_restored(self, tmp_path, output_dir):
        data = speckled((1, H, W))
        data[0, 10:14, 10:14] = NODATA
        data[0, 0, 0] = np.nan
        inp = write_tif(tmp_path / "a.tif", data)
        out = run(LeeFilterStep(), inp, output_dir, window_size=5)
        with rio.open(out) as src:
            got = src.read(1)
        bad = (data[0] == NODATA) | np.isnan(data[0])
        assert np.all(got[10:14, 10:14] == NODATA)
        assert got[0, 0] == NODATA
        expected = reference_lee(np.where(bad, 0, data[0]).astype("float64"), ~bad, 5, 4.4)
        np.testing.assert_allclose(got[~bad], expected[~bad], rtol=1e-4, atol=1e-6)

    def test_netcdf_stack_filters_each_variable_and_date(self, tmp_path, output_dir):
        vv, vh = speckled((3, H, W), 1), speckled((3, H, W), 2) * 0.3
        inp = write_stack(tmp_path / "stack_31TCJ.nc", {"VV": vv, "VH": vh})
        out = run(LeeFilterStep(), inp, output_dir, window_size=5)
        assert out.suffix == ".nc"
        ds = xr.open_dataset(out)
        assert set(ds.data_vars) == {"VV", "VH"} and ds.sizes["time"] == 3
        assert ds.attrs["tile"] == "31TCJ"
        for name, cube in (("VV", vv), ("VH", vh)):
            for t in range(3):
                expected = reference_lee(cube[t].astype("float64"), np.ones((H, W), bool), 5, 4.4)
                np.testing.assert_allclose(ds[name].isel(time=t).values, expected, rtol=1e-4, atol=1e-6)

    def test_netcdf_nodata_restored_and_attrs_kept(self, tmp_path, output_dir):
        cube = speckled((2, H, W))
        cube[1, 5:8, 5:8] = np.nan
        inp = write_stack(tmp_path / "s.nc", {"VV": cube})
        out = run(LeeFilterStep(), inp, output_dir)
        raw = xr.open_dataset(out, mask_and_scale=False)
        assert np.all(raw.VV.isel(time=1).values[5:8, 5:8] == NODATA)
        assert raw.VV.attrs["nodata"] == NODATA
        assert np.array_equal(raw.x.values, xr.open_dataset(inp).x.values)

    def test_netcdf_on_disk_chunks_match_processing_chunks(self, tmp_path, output_dir):
        inp = write_stack(tmp_path / "s.nc", {"VV": speckled((4, H, W))})
        out = run(LeeFilterStep(), inp, output_dir, window_size=3, memory_budget_mb=0.2)
        ds = xr.open_dataset(out)
        t, ch_y, ch_x = ds.VV.encoding["chunksizes"]
        assert t == 4 and ch_y < H and ch_x < W
        assert ds.VV.encoding["zlib"] is False

    def test_compress_flag(self, tmp_path, output_dir):
        inp = write_stack(tmp_path / "s.nc", {"VV": speckled((2, H, W))})
        out = run(LeeFilterStep(), inp, output_dir, compress=True)
        enc = xr.open_dataset(out).VV.encoding
        assert enc["zlib"] is True and enc["complevel"] == 1 and enc["shuffle"] is True

    def test_lazy_until_flush(self, tmp_path, output_dir):
        inp = write_tif(tmp_path / "a.tif", speckled((1, H, W)))
        pending = LeeFilterStep().execute([inp], output_dir)
        assert not list(output_dir.iterdir())
        pending.flush_all()
        assert (output_dir / "lee_filter_a.tif").exists()

    def test_db_units_refused(self, tmp_path, output_dir):
        nc = write_stack(tmp_path / "s.nc", {"VV": speckled((2, H, W))}, units="dB")
        with pytest.raises(ValueError, match="dB"):
            LeeFilterStep().execute([nc], output_dir)
        tif = write_tif(tmp_path / "a.tif", speckled((1, H, W)), units="dB")
        with pytest.raises(ValueError, match="dB"):
            LeeFilterStep().execute([tif], output_dir)

    @pytest.mark.parametrize("window", [2, 4, 1])
    def test_bad_window(self, tmp_path, output_dir, window):
        tif = write_tif(tmp_path / "a.tif", speckled((1, H, W)))
        with pytest.raises(ValueError, match="window_size"):
            LeeFilterStep().execute([tif], output_dir, window_size=window)

    def test_budget_too_small(self, tmp_path, output_dir):
        tif = write_tif(tmp_path / "a.tif", speckled((1, H, W)))
        with pytest.raises(ValueError, match="memory_budget_mb"):
            LeeFilterStep().execute([tif], output_dir, memory_budget_mb=0.0001)


# -- Quegan ---------------------------------------------------------------------------------


class TestQuegan:
    def test_matches_reference(self, tmp_path, output_dir):
        cube = speckled((5, H, W), 3)
        inp = write_stack(tmp_path / "s.nc", {"VV": cube})
        out = run(QueganFilterStep(), inp, output_dir, window_size=5)
        ds = xr.open_dataset(out)
        expected = reference_quegan(cube.astype("float64"), np.ones(cube.shape, bool), 5)
        np.testing.assert_allclose(ds.VV.values, expected, rtol=1e-4, atol=1e-7)
        assert list(ds.time.values) == list(xr.open_dataset(inp).time.values)

    def test_partial_validity_and_chunking(self, tmp_path, output_dir):
        cube = speckled((4, H, W), 4)
        cube[0, :, :20] = np.nan       # first date missing on the left half
        cube[2, 10:15, 10:15] = np.nan
        inp = write_stack(tmp_path / "s.nc", {"VV": cube})
        out = run(QueganFilterStep(), inp, output_dir, window_size=5, memory_budget_mb=0.2)
        got = xr.open_dataset(out).VV.values
        valid = ~np.isnan(cube)
        expected = reference_quegan(np.where(valid, cube, 0).astype("float64"), valid, 5)
        assert np.isnan(got[0, :, :20]).all() and np.isnan(got[2, 10:15, 10:15]).all()
        np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-4, atol=1e-7)

    def test_rejects_tif(self, tmp_path, output_dir):
        tif = write_tif(tmp_path / "a.tif", speckled((3, H, W)))
        with pytest.raises(ValueError, match="Tile stack"):
            QueganFilterStep().execute([tif], output_dir)

    def test_rejects_single_date(self, tmp_path, output_dir):
        nc = write_stack(tmp_path / "s.nc", {"VV": speckled((1, H, W))})
        with pytest.raises(ValueError, match="at least 2"):
            QueganFilterStep().execute([nc], output_dir)

    def test_rejects_db(self, tmp_path, output_dir):
        nc = write_stack(tmp_path / "s.nc", {"VV": speckled((3, H, W))}, units="dB")
        with pytest.raises(ValueError, match="dB"):
            QueganFilterStep().execute([nc], output_dir)


# -- Pipeline + planner ---------------------------------------------------------------------


def test_chain_via_composition(tmp_path):
    import eo_pipe
    from eo_pipe import ParallelBatch, PipelineComposition

    inp = write_stack(tmp_path / "stack_31TCJ.nc", {"VV": speckled((3, H, W)), "VH": speckled((3, H, W), 9)})
    ctx = (
        PipelineComposition(workspace=tmp_path / "ws")
        .add_step("lee_filter", ParallelBatch(), window_size=5, memory_budget_mb=1)
        .add_step("quegan_filter", ParallelBatch(), window_size=5, memory_budget_mb=1)
        .run(inputs=[inp], save_intermediate=True)
    )
    assert ctx.inputs[0].name == "quegan_filter_lee_filter_stack_31TCJ.nc"
    assert xr.open_dataset(ctx.inputs[0]).sizes["time"] == 3


class TestPlanFilterChunks:
    def test_shrinks_with_budget(self):
        big = plan_filter_chunks(16384, 100, 10980, 10980, 7)
        small = plan_filter_chunks(64, 100, 10980, 10980, 7)
        assert small.side < big.side

    def test_reduces_workers_before_chunk(self):
        roomy = plan_filter_chunks(4096, 100, 10980, 10980, 7, )
        tight = plan_filter_chunks(64, 100, 10980, 10980, 7)
        assert tight.workers <= roomy.workers

    def test_error_names_minimum(self):
        with pytest.raises(ValueError, match="at least"):
            plan_filter_chunks(0.01, 100, 10980, 10980, 7)

    def test_padded_chunk_stays_within_budget(self):
        from eo_pipe.io.memory import OVERHEAD
        from eo_pipe.io.speckle import _WORK_BYTES

        for window in (3, 7, 41):
            halo = window // 2
            plan = plan_filter_chunks(256, 100, 10980, 10980, window)
            padded = (plan.side + 2 * halo) ** 2
            assert plan.workers * padded * 100 * _WORK_BYTES * OVERHEAD <= 256 * 2**20

    def test_halo_larger_than_budget_errors(self):
        # a 16 px chunk plus a 201 px halo cannot fit 1 MB at 100 dates
        with pytest.raises(ValueError, match="at least"):
            plan_filter_chunks(1, 100, 10980, 10980, 401)


def test_budget_validated():
    with pytest.raises(ValueError):
        budget_mb({"memory_budget_mb": 0})
