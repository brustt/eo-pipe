"""Tests for eo_pipe.io.raster_io."""

from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

from eo_pipe.io.raster_io import (
    DEFAULT_WRITER,
    RasterWriter,
    _build_lut,
    _match_band_integer,
    downsample_raster,
    hist_match_worker,
)

# ---------------------------------------------------------------------------
# RasterWriter
# ---------------------------------------------------------------------------


class TestRasterWriter:
    def test_write_creates_file(self, write_raster, output_dir):
        inp = write_raster("src.tif")
        out = output_dir / "out.tif"

        with rio.open(inp) as src:
            data = src.read()
            profile = src.profile.copy()

        DEFAULT_WRITER.write(out, data, **profile)
        assert out.exists()

    def test_written_data_matches(self, write_raster, output_dir):
        inp = write_raster("src.tif")
        out = output_dir / "out.tif"

        with rio.open(inp) as src:
            data = src.read()
            profile = src.profile.copy()

        DEFAULT_WRITER.write(out, data, **profile)

        with rio.open(out) as dst:
            result = dst.read()

        np.testing.assert_array_equal(data, result)

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "out.tif"
        rng = np.random.default_rng(0)
        data = rng.integers(0, 255, (1, 16, 16), dtype=np.uint8)

        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        DEFAULT_WRITER.write(
            deep,
            data,
            driver="GTiff",
            dtype="uint8",
            width=16,
            height=16,
            count=1,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(0, 0, 1, 1, 16, 16),
        )
        assert deep.exists()

    def test_predictor_auto_integer(self):
        writer = RasterWriter()
        assert writer._resolve_predictor(np.dtype("uint8")) == 2

    def test_predictor_auto_float(self):
        writer = RasterWriter()
        assert writer._resolve_predictor(np.dtype("float32")) == 3

    def test_predictor_explicit_override(self):
        writer = RasterWriter(predictor=1)
        assert writer._resolve_predictor(np.dtype("uint8")) == 1

    def test_compression_none_no_compress_key(self):
        writer = RasterWriter(compress="none")
        opts = writer._creation_options(np.dtype("uint8"))
        assert "compress" not in opts

    def test_tiling_off_no_block_keys(self):
        writer = RasterWriter(tiled=False)
        opts = writer._creation_options(np.dtype("uint8"))
        assert "blockxsize" not in opts
        assert "blockysize" not in opts

    def test_extra_options_merged(self):
        writer = RasterWriter(extra={"INTERLEAVE": "PIXEL"})
        opts = writer._creation_options(np.dtype("uint8"))
        assert opts["INTERLEAVE"] == "PIXEL"


# ---------------------------------------------------------------------------
# downsample_raster
# ---------------------------------------------------------------------------


class TestDownsampleRaster:
    def test_downsample_by_factor(self, single_raster, output_dir):
        out = output_dir / "ds.tif"
        downsample_raster(single_raster, out, downsample_factor=2)

        with rio.open(out) as dst:
            assert dst.width == 32
            assert dst.height == 32

    def test_downsample_by_target_resolution(self, single_raster, output_dir):
        # source is 0.01 deg/px; target 0.02 → half the pixels
        out = output_dir / "ds_res.tif"
        downsample_raster(single_raster, out, target_resolution=0.02)

        with rio.open(out) as dst:
            assert dst.width == 32
            assert dst.height == 32

    def test_upsample_supported(self, single_raster, output_dir):
        out = output_dir / "up.tif"
        downsample_raster(single_raster, out, downsample_factor=0.5)

        with rio.open(out) as dst:
            assert dst.width == 128

    def test_returns_path(self, single_raster, output_dir):
        out = output_dir / "ret.tif"
        result = downsample_raster(single_raster, out, downsample_factor=2)
        assert isinstance(result, Path)
        assert result == out

    def test_both_params_raises(self, single_raster, output_dir):
        with pytest.raises(ValueError, match="not both"):
            downsample_raster(
                single_raster,
                output_dir / "x.tif",
                downsample_factor=2,
                target_resolution=0.02,
            )

    def test_neither_param_raises(self, single_raster, output_dir):
        with pytest.raises(ValueError, match="must be provided"):
            downsample_raster(single_raster, output_dir / "x.tif")

    def test_invalid_method_raises(self, single_raster, output_dir):
        with pytest.raises(ValueError, match="not known"):
            downsample_raster(
                single_raster, output_dir / "x.tif",
                downsample_factor=2, method="unicorn"
            )

    def test_nodata_written(self, single_raster, output_dir):
        out = output_dir / "nd.tif"
        downsample_raster(single_raster, out, downsample_factor=2, nodata_value=255)
        with rio.open(out) as dst:
            assert dst.nodata == 255


# ---------------------------------------------------------------------------
# Histogram matching internals
# ---------------------------------------------------------------------------


class TestBuildLut:
    def test_identity_same_image(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 256, 1000, dtype=np.uint8)
        lut = _build_lut(data, data)
        # LUT applied to source should not change values drastically
        mapped = lut[data]
        # Mean should be preserved approximately
        assert abs(int(mapped.mean()) - int(data.mean())) < 10

    def test_lut_length(self):
        data = np.arange(256, dtype=np.uint8)
        lut = _build_lut(data, data)
        assert lut.shape == (256,)

    def test_lut_dtype_matches_input(self):
        data = np.arange(256, dtype=np.uint8)
        lut = _build_lut(data, data)
        assert lut.dtype == np.uint8

    def test_monotone_shift(self):
        """Matching a bright reference shifts the mean up."""
        rng = np.random.default_rng(1)
        dark = rng.integers(0, 100, 10000, dtype=np.uint8)
        bright = rng.integers(155, 256, 10000, dtype=np.uint8)
        lut = _build_lut(dark, bright)
        assert lut[50] > 50  # dark value mapped higher


class TestMatchBandInteger:
    def test_full_match_proportion(self):
        rng = np.random.default_rng(2)
        src = rng.integers(0, 100, (64, 64), dtype=np.uint8)
        ref = rng.integers(150, 256, (64, 64), dtype=np.uint8)
        matched = _match_band_integer(src, ref, None, 1.0)
        assert matched.dtype == np.uint8
        assert matched.mean() > src.mean()

    def test_zero_proportion_unchanged(self):
        rng = np.random.default_rng(3)
        src = rng.integers(0, 100, (64, 64), dtype=np.uint8)
        ref = rng.integers(150, 256, (64, 64), dtype=np.uint8)
        matched = _match_band_integer(src, ref, None, 0.0)
        np.testing.assert_array_equal(matched, src)

    def test_valid_mask_excludes_nodata(self):
        rng = np.random.default_rng(4)
        src = rng.integers(10, 200, (64, 64), dtype=np.uint8)
        ref = rng.integers(10, 200, (64, 64), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=bool)
        mask[0, :] = False  # top row is nodata
        # Should not raise
        matched = _match_band_integer(src, ref, mask, 1.0)
        assert matched.shape == (64, 64)


# ---------------------------------------------------------------------------
# hist_match_worker (end-to-end)
# ---------------------------------------------------------------------------


class TestHistMatchWorker:
    def test_output_file_created(self, single_raster, nodata_raster, output_dir):
        dst = output_dir / "matched.tif"
        result = hist_match_worker(single_raster, nodata_raster, dst)
        assert result == dst
        assert dst.exists()

    def test_output_dtype_preserved(self, single_raster, nodata_raster, output_dir):
        dst = output_dir / "matched.tif"
        hist_match_worker(single_raster, nodata_raster, dst)
        with rio.open(dst) as f:
            assert f.dtypes[0] == "uint8"

    def test_nodata_pixels_restored(self, write_raster, output_dir):
        """Nodata pixels in the source should be unchanged in the output."""
        rng = np.random.default_rng(55)
        src_data = rng.integers(10, 200, (3, 64, 64), dtype=np.uint8)
        src_data[:, 0, :] = 0  # nodata row

        ref_data = rng.integers(100, 240, (3, 64, 64), dtype=np.uint8)

        src = write_raster("src_nd.tif", data=src_data)
        ref = write_raster("ref_nd.tif", data=ref_data)
        dst = output_dir / "matched_nd.tif"
        hist_match_worker(src, ref, dst)

        with rio.open(dst) as f:
            result = f.read()
        # nodata row should still be 0
        np.testing.assert_array_equal(result[:, 0, :], 0)

    def test_proportion_blending(self, write_raster, output_dir):
        """proportion=0 should return data close to the source."""
        rng = np.random.default_rng(77)
        src_data = rng.integers(10, 100, (3, 64, 64), dtype=np.uint8)
        ref_data = rng.integers(200, 255, (3, 64, 64), dtype=np.uint8)

        src = write_raster("src_blend.tif", data=src_data)
        ref = write_raster("ref_blend.tif", data=ref_data)

        dst0 = output_dir / "blend0.tif"
        hist_match_worker(src, ref, dst0, match_proportion=0.0)

        with rio.open(dst0) as f:
            result = f.read()
        np.testing.assert_array_equal(result, src_data)

    def test_save_false_no_file(self, single_raster, nodata_raster, output_dir):
        dst = output_dir / "never.tif"
        hist_match_worker(single_raster, nodata_raster, dst, save=False)
        assert not dst.exists()
