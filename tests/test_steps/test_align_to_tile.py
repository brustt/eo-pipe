"""Tests for AlignToTileStep — behaviour observed on the files it writes."""

from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from eo_pipe.steps.raster.align_tile import AlignToTileStep

# Origin of the synthetic tile 31TCJ (real Sentinel-2 granule origin)
TILE_UL = (300000.0, 4900020.0)


def make_raster(
    path: Path,
    x0: float,
    y0: float,
    *,
    res: float = 10.0,
    size: int = 64,
    crs: str = "EPSG:32631",
    count: int = 1,
    dtype: str = "float32",
    nodata: float = 0.0,
    fill=None,
    descriptions=None,
    tags=None,
) -> Path:
    rng = np.random.default_rng(3)
    data = rng.uniform(1, 100, (count, size, size)).astype(dtype) if fill is None else fill
    with rio.open(
        path, "w", driver="GTiff", width=size, height=size, count=count, dtype=dtype,
        crs=crs, transform=from_origin(x0, y0, res, res), nodata=nodata,
    ) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = descriptions
        if tags:
            dst.update_tags(**tags)
    return path


def run(inputs, output_dir, grid, **params):
    return AlignToTileStep().execute(list(inputs), output_dir, grid_path=grid, **params).flush_all()


def whole_pixels(a: float, b: float, res: float = 10.0) -> bool:
    return abs((a - b) / res - round((a - b) / res)) < 1e-6


@pytest.fixture()
def scene(tmp_path):
    """On-lattice 10 m scene fully inside tile 31TCJ."""
    return make_raster(tmp_path / "S1A_20250314T051233_vv.tif", 310000.0, 4890000.0)


class TestOutputsAndNaming:
    def test_one_output_per_tile_with_named_tokens(self, scene, output_dir, tile_grid):
        result = run([scene], output_dir, tile_grid)
        assert [p.name for p in result.outputs] == [
            "align_31TCJ_20250314T051233_S1A_20250314T051233_vv.tif"
        ]

    def test_output_is_in_the_tile_crs(self, scene, output_dir, tile_grid):
        out = run([scene], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            assert dst.crs == CRS.from_epsg(32631)

    def test_tags_identify_tile_date_source_polarization(self, scene, output_dir, tile_grid):
        out = run([scene], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            tags = dst.tags()
        assert tags["TILE"] == "31TCJ"
        assert tags["DATETIME"] == "2025-03-14T05:12:33"
        assert tags["SOURCE"] == "S1A_20250314T051233_vv"
        assert tags["POLARIZATION"] == "VV"

    def test_input_outside_grid_gives_no_output(self, tmp_path, output_dir, tile_grid):
        far = make_raster(tmp_path / "x_20250101.tif", 500000.0, 100000.0)
        assert run([far], output_dir, tile_grid).outputs == []


class TestLattice:
    def test_on_lattice_input_is_copied_without_resampling(self, scene, output_dir, tile_grid):
        out = run([scene], output_dir, tile_grid).outputs[0]
        with rio.open(scene) as src, rio.open(out) as dst:
            assert (dst.width, dst.height) == (64, 64)
            assert dst.transform.c == src.transform.c and dst.transform.f == src.transform.f
            np.testing.assert_array_equal(dst.read(), src.read())

    def test_off_lattice_input_lands_on_whole_pixel_offsets(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "a_20250314.tif", 310003.7, 4890004.3)
        out = run([src], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            assert whole_pixels(dst.transform.c, TILE_UL[0])
            assert whole_pixels(dst.transform.f, TILE_UL[1])
            assert dst.res == (10.0, 10.0)
            assert (dst.read(1) != 0).any()

    def test_repeat_pass_crops_differ_by_whole_pixels(self, tmp_path, output_dir, tile_grid):
        a = make_raster(tmp_path / "a_20250314.tif", 310000.0, 4890000.0)
        b = make_raster(tmp_path / "b_20250326.tif", 310037.0, 4890011.0)
        oa, ob = run([a, b], output_dir, tile_grid).outputs
        with rio.open(oa) as da, rio.open(ob) as db:
            assert whole_pixels(da.transform.c, db.transform.c)
            assert whole_pixels(da.transform.f, db.transform.f)

    def test_resolution_parameter_sets_pixel_size(self, scene, output_dir, tile_grid):
        out = run([scene], output_dir, tile_grid, resolution=20.0).outputs[0]
        with rio.open(out) as dst:
            assert dst.res == (20.0, 20.0)
            assert whole_pixels(dst.transform.c, TILE_UL[0], 20.0)

    def test_resolution_must_divide_the_tile(self, scene, output_dir, tile_grid):
        with pytest.raises(ValueError, match="resolution"):
            run([scene], output_dir, tile_grid, resolution=7.0)

    def test_geographic_input_is_reprojected_to_the_tile_utm(self, tmp_path, output_dir, tile_grid):
        src = make_raster(
            tmp_path / "g_20250314.tif", 6.5, 45.5, res=0.0005, crs="EPSG:4326"
        )
        out = run([src], output_dir, tile_grid).outputs[0]
        assert out.name.startswith("align_32TLR_")
        with rio.open(out) as dst:
            assert dst.crs == CRS.from_epsg(32632)
            assert (dst.read(1) != 0).any()


class TestSplitAcrossTiles:
    def test_raster_in_tile_overlap_gives_one_output_per_tile(self, tmp_path, output_dir, tile_grid):
        # x 409400..410040 lies inside 31TCJ (to 409800) and 31TDJ (from 399960)
        src = make_raster(tmp_path / "s_20250314.tif", 409400.0, 4890000.0)
        outs = run([src], output_dir, tile_grid).outputs
        assert sorted(p.name.split("_")[1] for p in outs) == ["31TCJ", "31TDJ"]
        with rio.open(next(p for p in outs if "_31TCJ_" in p.name)) as dst:
            assert dst.bounds.right <= 409800.0
            assert dst.width == 40
        with rio.open(next(p for p in outs if "_31TDJ_" in p.name)) as dst:
            assert dst.width == 64
            assert whole_pixels(dst.transform.c, 399960.0)


class TestExtent:
    def test_full_tile_writes_the_whole_tile(self, scene, output_dir, tile_grid):
        out = run([scene], output_dir, tile_grid, resolution=60.0, full_tile=True).outputs[0]
        with rio.open(out) as dst:
            assert (dst.width, dst.height) == (1830, 1830)
            assert dst.bounds == (300000.0, 4790220.0, 409800.0, 4900020.0)

    def test_aoi_gives_identical_extent_to_every_date(self, tmp_path, output_dir, tile_grid):
        a = make_raster(tmp_path / "a_20250314.tif", 310000.0, 4890000.0)
        b = make_raster(tmp_path / "b_20250326.tif", 310037.0, 4890011.0)
        # AOI in EPSG:32631 covering most of the overlap of a and b
        aoi = box(310100.0, 4889500.0, 310500.0, 4889900.0)
        oa, ob = run([a, b], output_dir, tile_grid, aoi=aoi, aoi_crs="EPSG:32631").outputs
        with rio.open(oa) as da, rio.open(ob) as db:
            assert da.bounds == db.bounds
            assert (da.width, da.height) == (db.width, db.height)
            assert da.bounds.left <= 310100.0 and da.bounds.right >= 310500.0

    def test_aoi_and_full_tile_conflict(self, scene, output_dir, tile_grid):
        with pytest.raises(ValueError, match="aoi"):
            run([scene], output_dir, tile_grid, aoi=box(0, 0, 1, 1), full_tile=True)


class TestValidity:
    def test_all_nodata_window_is_skipped(self, tmp_path, output_dir, tile_grid):
        empty = make_raster(
            tmp_path / "e_20250314.tif", 310000.0, 4890000.0, fill=np.zeros((1, 64, 64), "float32")
        )
        assert run([empty], output_dir, tile_grid).outputs == []

    def test_min_valid_fraction_skips_sparse_windows(self, tmp_path, output_dir, tile_grid):
        data = np.zeros((1, 64, 64), "float32")
        data[0, :4, :4] = 5.0   # 16 / 4096 valid
        sparse = make_raster(tmp_path / "s_20250314.tif", 310000.0, 4890000.0, fill=data)
        assert len(run([sparse], output_dir, tile_grid).outputs) == 1
        assert run([sparse], output_dir, tile_grid, min_valid_fraction=0.5).outputs == []


class TestDatetimeAndPolarization:
    def test_missing_date_raises_naming_the_file(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "nodate_vv.tif", 310000.0, 4890000.0)
        with pytest.raises(ValueError, match="nodate_vv"):
            run([src], output_dir, tile_grid)

    def test_date_only_stem_uses_eight_character_token(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "scene_20250314_vh.tif", 310000.0, 4890000.0)
        out = run([src], output_dir, tile_grid).outputs[0]
        assert out.name == "align_31TCJ_20250314_scene_20250314_vh.tif"
        with rio.open(out) as dst:
            assert dst.tags()["DATETIME"] == "2025-03-14"
            assert dst.tags()["POLARIZATION"] == "VH"

    def test_date_pattern_override(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "scene_2025-03-14.tif", 310000.0, 4890000.0)
        out = run([src], output_dir, tile_grid, date_pattern=r"(\d{4})-(\d{2})-(\d{2})").outputs[0]
        assert "_20250314_" in out.name

    def test_polarization_from_band_description(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "s_20250314.tif", 310000.0, 4890000.0, descriptions=["vh"])
        out = run([src], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            assert dst.tags()["POLARIZATION"] == "VH"

    def test_polarization_override(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "s_20250314_vv.tif", 310000.0, 4890000.0)
        out = run([src], output_dir, tile_grid, polarization="HH").outputs[0]
        with rio.open(out) as dst:
            assert dst.tags()["POLARIZATION"] == "HH"

    def test_unknown_polarization_leaves_tag_absent(self, tmp_path, output_dir, tile_grid):
        src = make_raster(tmp_path / "s_20250314.tif", 310000.0, 4890000.0)
        out = run([src], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            assert "POLARIZATION" not in dst.tags()


class TestMultiBand:
    def test_band_descriptions_and_source_tags_are_kept(self, tmp_path, output_dir, tile_grid):
        src = make_raster(
            tmp_path / "s_20250314.tif", 310000.0, 4890000.0, count=2,
            descriptions=["VV", "VH"], tags={"orbit": "ascending"},
        )
        out = run([src], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            assert dst.count == 2
            assert dst.descriptions == ("VV", "VH")
            assert dst.tags()["orbit"] == "ascending"
            assert "POLARIZATION" not in dst.tags()


class TestRegistration:
    def test_registered_by_name(self):
        from eo_pipe.pipeline.registry import StepRegistry

        assert AlignToTileStep.name == "align_to_tile"
        assert "align_to_tile" in StepRegistry.available()


class TestSourceZerosAreNotNodata:
    def test_zeros_survive_resampling_when_source_has_no_nodata(self, tmp_path, output_dir, tile_grid):
        data = np.zeros((1, 64, 64), "float32")
        data[0, 20:40, 20:40] = 5.0
        src = tmp_path / "z_20250314.tif"
        with rio.open(
            src, "w", driver="GTiff", width=64, height=64, count=1, dtype="float32",
            crs="EPSG:32631", transform=from_origin(310003.0, 4890004.0, 10.0, 10.0),
        ) as dst:
            dst.write(data)   # no nodata declared
        out = run([src], output_dir, tile_grid, nodata=-9999.0).outputs[0]
        with rio.open(out) as dst:
            arr = dst.read(1)
        assert (arr == 0.0).sum() > 1000          # real zeros kept as values
        assert (arr == -9999.0).sum() < arr.size  # only the padding is nodata


class TestNodataPropagation:
    def test_nodata_pixels_stay_nodata_after_warp(self, tmp_path, output_dir, tile_grid):
        data = np.full((1, 64, 64), 5.0, "float32")
        data[0, :, :32] = 0.0   # left half nodata (nodata=0 by default in make_raster)
        src = make_raster(tmp_path / "n_20250314.tif", 310003.0, 4890004.0, fill=data)
        out = run([src], output_dir, tile_grid).outputs[0]
        with rio.open(out) as dst:
            arr = dst.read(1)
        assert (arr[:, :28] == 0).all() and (arr[2:-2, 38:-3] == 5.0).all()


class TestMemoryBudget:
    """Aligned rasters are produced window by window; content must not depend on the budget."""

    TINY = 0.4  # MB: ~30 pixel rows per window

    @staticmethod
    def _read(path: Path):
        with rio.open(path) as dst:
            return dst.read(), dst.transform, dst.tags(), dst.descriptions, dst.nodata

    def _same(self, a: Path, b: Path) -> None:
        da, ta, tags_a, desc_a, nd_a = self._read(a)
        db, tb, tags_b, desc_b, nd_b = self._read(b)
        np.testing.assert_array_equal(da, db)
        assert ta == tb and tags_a == tags_b and desc_a == desc_b and nd_a == nd_b

    def test_warped_windows_equal_the_single_window_result(self, tmp_path, tile_grid):
        src = make_raster(
            tmp_path / "g_20250314.tif", 6.5, 45.5, res=0.0005, size=200, crs="EPSG:4326",
            count=2, descriptions=("a", "b"),
        )
        for method in ("bilinear", "cubic"):
            big = run([src], tmp_path / f"big_{method}", tile_grid, method=method).outputs[0]
            small = run([src], tmp_path / f"small_{method}", tile_grid, method=method, memory_budget_mb=self.TINY).outputs[0]
            self._same(big, small)

    def test_lattice_windows_equal_the_single_window_result(self, scene, tmp_path, tile_grid):
        big = run([scene], tmp_path / "big", tile_grid, full_tile=True, resolution=100.0).outputs[0]
        small = run([scene], tmp_path / "small", tile_grid, full_tile=True, resolution=100.0, memory_budget_mb=self.TINY).outputs[0]
        self._same(big, small)

    def test_input_spanning_several_tiles_gives_the_same_outputs(self, tmp_path, tile_grid):
        src = make_raster(tmp_path / "s_20250314_vv.tif", 409400.0, 4890000.0)
        big = run([src], tmp_path / "big", tile_grid).outputs
        small = run([src], tmp_path / "small", tile_grid, memory_budget_mb=self.TINY).outputs
        assert len(big) == len(small) == 2
        for a, b in zip(big, small):
            assert a.name == b.name
            self._same(a, b)

    def test_skipped_tile_leaves_no_file_with_small_budget(self, tmp_path, output_dir, tile_grid):
        empty = make_raster(tmp_path / "e_20250314_vv.tif", 310000.0, 4890000.0, fill=np.zeros((1, 64, 64), "float32"))
        assert run([empty], output_dir, tile_grid, memory_budget_mb=self.TINY).outputs == []
        assert not list(output_dir.glob("*"))

    def test_sparse_window_is_skipped_by_min_valid_fraction_across_windows(self, tmp_path, output_dir, tile_grid):
        sparse = np.zeros((1, 64, 64), "float32")
        sparse[0, 0, :4] = 5.0
        src = make_raster(tmp_path / "p_20250314_vv.tif", 310000.0, 4890000.0, fill=sparse)
        assert run([src], output_dir, tile_grid, min_valid_fraction=0.5, memory_budget_mb=self.TINY).outputs == []
        assert len(run([src], output_dir, tile_grid, memory_budget_mb=self.TINY).outputs) == 1

    def test_full_tile_peak_allocation_is_bounded(self, scene, output_dir, tile_grid):
        import tracemalloc

        # full Tile at 100 m: 1098 x 1098 px x 2 bands x 4 B = 9.6 MB if held whole
        two_band = make_raster(
            scene.parent / "t_20250314_vv.tif", 310000.0, 4890000.0, count=2, size=1000, res=100.0,
            fill=np.full((2, 1000, 1000), 3, "float32"),
        )
        tracemalloc.start()
        try:
            AlignToTileStep().execute(
                [two_band], output_dir, grid_path=tile_grid, full_tile=True, resolution=100.0, memory_budget_mb=1,
            ).flush_all()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 5 * 2**20

    def test_budget_too_small_for_one_row_raises(self, scene, output_dir, tile_grid):
        with pytest.raises(ValueError, match=r"memory_budget_mb.*at least"):
            AlignToTileStep().execute([scene], output_dir, grid_path=tile_grid, memory_budget_mb=1e-6)
