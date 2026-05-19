"""Tests for OTBStepBase, OrthoRectifyStep, SARCalibrationStep, SARBorderCutStep."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.pipeline.base import StepOutput
from eo_pipe.pipeline.registry import StepRegistry
from eo_pipe.steps.otb.base import OTBStepBase
from eo_pipe.steps.otb.orthorectify import OrthoRectifyStep, _map_projection_params
from eo_pipe.steps.otb.sar_border_cut import SARBorderCutStep, _detect_s1_borders
from eo_pipe.steps.otb.sar_calibrate import SARCalibrationStep

_FAKE_CLI = "/usr/bin/otbcli_OrthoRectification"
_OTB_APP = "eo_pipe.steps.otb.base"


def _mock_proc(returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stderr = stderr
    return m

W, H = 64, 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raster(path: Path, crs: CRS, bounds: tuple = (0.0, 0.0, 0.64, 0.64)) -> Path:
    transform = from_bounds(*bounds, W, H)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        path, "w",
        driver="GTiff", dtype="uint8", width=W, height=H,
        count=1, crs=crs, transform=transform,
    ) as dst:
        dst.write(np.ones((1, H, W), dtype=np.uint8))
    return path


def _geographic_raster(tmp_path: Path, name: str = "img.tif") -> Path:
    return _write_raster(tmp_path / name, CRS.from_epsg(4326))


def _utm31n_raster(tmp_path: Path, name: str = "img.tif") -> Path:
    return _write_raster(
        tmp_path / name,
        CRS.from_epsg(32631),
        bounds=(300000.0, 5000000.0, 306400.0, 5006400.0),
    )


# ---------------------------------------------------------------------------
# _map_projection_params
# ---------------------------------------------------------------------------


class TestMapProjectionParams:
    def test_geographic_returns_wgs(self):
        crs = CRS.from_epsg(4326)
        assert _map_projection_params(crs) == {"map": "wgs"}

    def test_utm_north_zone31(self):
        crs = CRS.from_epsg(32631)
        result = _map_projection_params(crs)
        assert result == {"map": "utm", "map.utm.zone": 31, "map.utm.northhem": True}

    def test_utm_south_zone31(self):
        crs = CRS.from_epsg(32731)
        result = _map_projection_params(crs)
        assert result == {"map": "utm", "map.utm.zone": 31, "map.utm.northhem": False}

    def test_utm_north_zone1_boundary(self):
        crs = CRS.from_epsg(32601)
        result = _map_projection_params(crs)
        assert result["map"] == "utm"
        assert result["map.utm.zone"] == 1
        assert result["map.utm.northhem"] is True

    def test_utm_south_zone60_boundary(self):
        crs = CRS.from_epsg(32760)
        result = _map_projection_params(crs)
        assert result["map"] == "utm"
        assert result["map.utm.zone"] == 60
        assert result["map.utm.northhem"] is False

    def test_other_epsg_uses_epsg_map(self):
        crs = CRS.from_epsg(2154)  # RGF93 / Lambert-93
        result = _map_projection_params(crs)
        assert result == {"map": "epsg", "map.epsg.code": 2154}

    def test_no_epsg_raises_value_error(self):
        crs = MagicMock()
        crs.is_geographic = False
        crs.to_epsg.return_value = None
        crs.to_wkt.return_value = "PROJCRS[...]"
        with pytest.raises(ValueError, match="Cannot determine OTB projection"):
            _map_projection_params(crs)


# ---------------------------------------------------------------------------
# OTBStepBase — abstract contract
# ---------------------------------------------------------------------------


class TestOTBStepBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            OTBStepBase()  # abstract build_otb_params

    def test_concrete_without_otb_app_raises_at_execute(self, tmp_path):
        class _NoAppStep(OTBStepBase):
            name = "_no_app"

            def build_otb_params(self, inputs, output_path, **params):
                return {}

        src = _geographic_raster(tmp_path, "src.tif")
        step = _NoAppStep()
        step.__class__.otb_app = ""
        with pytest.raises(TypeError, match="must define an 'otb_app'"):
            step.execute([src], tmp_path / "out")


# ---------------------------------------------------------------------------
# OrthoRectifyStep — class contract
# ---------------------------------------------------------------------------


class TestOrthoRectifyStepContract:
    def test_registered_in_step_registry(self):
        assert StepRegistry._registry.get("orthorectify") is OrthoRectifyStep

    def test_class_vars(self):
        assert OrthoRectifyStep.name == "orthorectify"
        assert OrthoRectifyStep.otb_app == "OrthoRectification"
        assert OrthoRectifyStep.param_in == "io.in"
        assert OrthoRectifyStep.param_out == "io.out"


# ---------------------------------------------------------------------------
# OrthoRectifyStep — build_otb_params
# ---------------------------------------------------------------------------


class TestBuildOtbParams:
    def test_geographic_ref_sets_wgs_map(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)

        assert params["map"] == "wgs"
        assert params["io.in"] == str(src)
        assert params["interpolator"] == "bco"
        assert params["outputs.sizex"] == W
        assert params["outputs.sizey"] == H
        assert "io.out" not in params  # injected by base class

    def test_utm_ref_sets_utm_map(self, tmp_path):
        ref = _utm31n_raster(tmp_path, "ref.tif")
        src = _utm31n_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)

        assert params["map"] == "utm"
        assert params["map.utm.zone"] == 31
        assert params["map.utm.northhem"] is True

    def test_custom_interpolator(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, interpolator="nn"
        )
        assert params["interpolator"] == "nn"

    def test_elev_dem_included_when_set(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        dem_dir = tmp_path / "dem"
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, elev_dem=dem_dir
        )
        assert params["elev.dem"] == str(dem_dir)
        assert "elev.geoid" not in params

    def test_elev_geoid_included_when_set(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        geoid = tmp_path / "egm96.bsb"
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, elev_geoid=geoid
        )
        assert params["elev.geoid"] == str(geoid)
        assert "elev.dem" not in params

    def test_both_elev_params(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        dem_dir = tmp_path / "dem"
        geoid = tmp_path / "egm96.bsb"
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, elev_dem=dem_dir, elev_geoid=geoid
        )
        assert "elev.dem" in params
        assert "elev.geoid" in params

    def test_no_elev_params_by_default(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)
        assert "elev.dem" not in params
        assert "elev.geoid" not in params

    def test_grid_spacing_and_ram(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, grid_spacing=2.0, ram_mb=512
        )
        assert params["opt.gridspacing"] == 2.0
        assert params["opt.ram"] == 512

    def test_output_size_matches_ref_when_fully_covered(self, tmp_path):
        ref = _utm31n_raster(tmp_path, "ref.tif")
        src = _utm31n_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)
        assert params["outputs.sizex"] == W
        assert params["outputs.sizey"] == H

    def test_output_extent_clipped_to_source_footprint(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")  # bounds (0, 0, 0.64, 0.64)
        # Source covers only left half: (0, 0, 0.32, 0.64)
        src = _write_raster(tmp_path / "src.tif", CRS.from_epsg(4326), bounds=(0.0, 0.0, 0.32, 0.64))
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)
        assert params["outputs.sizex"] == W // 2
        assert params["outputs.sizey"] == H
        assert params["outputs.ulx"] == pytest.approx(0.0)

    def test_no_overlap_raises_value_error(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")  # bounds (0, 0, 0.64, 0.64)
        src = _write_raster(tmp_path / "src.tif", CRS.from_epsg(4326), bounds=(1.0, 1.0, 1.64, 1.64))
        with pytest.raises(ValueError, match="no spatial overlap"):
            OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)

    def test_multiple_inputs_raises_value_error(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src1 = _geographic_raster(tmp_path, "src1.tif")
        src2 = _geographic_raster(tmp_path, "src2.tif")
        with pytest.raises(ValueError, match="ParallelBatch"):
            OrthoRectifyStep().build_otb_params([src1, src2], tmp_path / "out.tif", ref=ref)

    def test_missing_ref_raises_type_error(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        with pytest.raises(TypeError, match="ref=Path"):
            OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif")

    def test_pixel_spacing_positive_x_negative_y(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)
        assert params["outputs.spacingx"] > 0
        assert params["outputs.spacingy"] < 0

    def test_extra_params_ignored(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        # Should not raise
        OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, unknown_param="whatever"
        )


# ---------------------------------------------------------------------------
# OrthoRectifyStep — execute() with mocked OTB
# ---------------------------------------------------------------------------


class TestOrthoRectifyExecute:
    def test_execute_calls_otb_cli(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "out"

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            OrthoRectifyStep().execute([src], out_dir, ref=ref)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "otbcli_OrthoRectification"

    def test_execute_injects_output_path(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "out"

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            OrthoRectifyStep().execute([src], out_dir, ref=ref)

        cmd = mock_run.call_args[0][0]
        io_out_idx = cmd.index("-io.out")
        # default compress=True → suffix after .tif
        assert "orthorectify_src.tif" in cmd[io_out_idx + 1]

    def test_compress_default_appends_gdal_suffix(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "out"

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            OrthoRectifyStep().execute([src], out_dir, ref=ref)

        cmd = mock_run.call_args[0][0]
        io_out_val = cmd[cmd.index("-io.out") + 1]
        assert "gdal:co:COMPRESS=DEFLATE" in io_out_val
        assert "gdal:co:TILED=YES" in io_out_val

    def test_compress_false_passes_plain_path(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "out"

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            OrthoRectifyStep().execute([src], out_dir, ref=ref, compress=False)

        cmd = mock_run.call_args[0][0]
        io_out_val = cmd[cmd.index("-io.out") + 1]
        assert io_out_val.endswith("orthorectify_src.tif")

    def test_execute_returns_step_output(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()):
            result = OrthoRectifyStep().execute([src], tmp_path / "out", ref=ref)

        assert isinstance(result, StepOutput)
        assert len(result.outputs) == 1
        assert isinstance(result.outputs[0], FlushedOutput)

    def test_execute_output_path_uses_prefixed_strategy(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "out"

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()):
            result = OrthoRectifyStep().execute([src], out_dir, ref=ref)

        assert result.outputs[0].path == out_dir / "orthorectify_src.tif"

    def test_execute_missing_cli_raises_runtime_error(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")

        with patch(f"{_OTB_APP}.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="not found on PATH"):
                OrthoRectifyStep().execute([src], tmp_path / "out", ref=ref)

    def test_execute_nonzero_exit_raises_runtime_error(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc(1, "OTB error")):
            with pytest.raises(RuntimeError, match="failed"):
                OrthoRectifyStep().execute([src], tmp_path / "out", ref=ref)

    def test_execute_creates_output_dir(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "deep" / "nested" / "out"

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()):
            OrthoRectifyStep().execute([src], out_dir, ref=ref)

        assert out_dir.exists()

    def test_flush_all_returns_step_result(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")

        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()):
            step_output = OrthoRectifyStep().execute([src], tmp_path / "out", ref=ref)

        result = step_output.flush_all()
        assert len(result.outputs) == 1
        assert result.outputs[0] == tmp_path / "out" / "orthorectify_src.tif"


# ---------------------------------------------------------------------------
# OrthoRectifyStep — orthofit mode
# ---------------------------------------------------------------------------


class TestOrthoRectifyOrthofit:
    def test_orthofit_sets_mode_and_ortho_param(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, ref_mode="orthofit"
        )
        assert params["outputs.mode"] == "orthofit"
        assert params["outputs.ortho"] == str(ref)

    def test_orthofit_omits_explicit_grid_params(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, ref_mode="orthofit"
        )
        for key in ("outputs.ulx", "outputs.uly", "outputs.sizex", "outputs.sizey",
                    "outputs.spacingx", "outputs.spacingy"):
            assert key not in params

    def test_orthofit_omits_map_projection_params(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, ref_mode="orthofit"
        )
        assert "map" not in params

    def test_orthofit_keeps_common_params(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, ref_mode="orthofit",
            interpolator="nn", grid_spacing=2.0, ram_mb=512,
        )
        assert params["interpolator"] == "nn"
        assert params["opt.gridspacing"] == 2.0
        assert params["opt.ram"] == 512

    def test_orthofit_elev_params_forwarded(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        dem = tmp_path / "dem"
        params = OrthoRectifyStep().build_otb_params(
            [src], tmp_path / "out.tif", ref=ref, ref_mode="orthofit", elev_dem=dem
        )
        assert params["elev.dem"] == str(dem)

    def test_invalid_ref_mode_raises(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        with pytest.raises(ValueError, match="ref_mode"):
            OrthoRectifyStep().build_otb_params(
                [src], tmp_path / "out.tif", ref=ref, ref_mode="bad"
            )

    def test_explicit_mode_default_unchanged(self, tmp_path):
        ref = _geographic_raster(tmp_path, "ref.tif")
        src = _geographic_raster(tmp_path, "src.tif")
        params = OrthoRectifyStep().build_otb_params([src], tmp_path / "out.tif", ref=ref)
        # explicit mode: no orthofit keys
        assert "outputs.mode" not in params
        assert "outputs.ortho" not in params
        # explicit mode: grid keys present
        assert "outputs.ulx" in params
        assert "outputs.sizex" in params


# ---------------------------------------------------------------------------
# SARCalibrationStep — class contract
# ---------------------------------------------------------------------------


class TestSARCalibrationStepContract:
    def test_registered_in_step_registry(self):
        assert StepRegistry._registry.get("sar_calibrate") is SARCalibrationStep

    def test_class_vars(self):
        assert SARCalibrationStep.name == "sar_calibrate"
        assert SARCalibrationStep.otb_app == "SARCalibration"
        assert SARCalibrationStep.param_in == "in"
        assert SARCalibrationStep.param_out == "out"


# ---------------------------------------------------------------------------
# SARCalibrationStep — build_otb_params
# ---------------------------------------------------------------------------


class TestSARCalibrationBuildParams:
    def test_defaults(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        params = SARCalibrationStep().build_otb_params([src], tmp_path / "out.tif")
        assert params["lut"] == "sigma"
        assert params["removenoise"] is False
        assert params["opt.ram"] == 256
        assert params["in"] == str(src)
        assert "out" not in params

    def test_custom_lut_gamma(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        params = SARCalibrationStep().build_otb_params([src], tmp_path / "out.tif", lut="gamma")
        assert params["lut"] == "gamma"

    def test_custom_lut_beta(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        params = SARCalibrationStep().build_otb_params([src], tmp_path / "out.tif", lut="beta")
        assert params["lut"] == "beta"

    def test_custom_lut_dn(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        params = SARCalibrationStep().build_otb_params([src], tmp_path / "out.tif", lut="dn")
        assert params["lut"] == "dn"

    def test_invalid_lut_raises(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        with pytest.raises(ValueError, match="Invalid lut"):
            SARCalibrationStep().build_otb_params([src], tmp_path / "out.tif", lut="bad")

    def test_removenoise_true(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        params = SARCalibrationStep().build_otb_params(
            [src], tmp_path / "out.tif", removenoise=True
        )
        assert params["removenoise"] is True

    def test_custom_ram(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        params = SARCalibrationStep().build_otb_params(
            [src], tmp_path / "out.tif", ram_mb=512
        )
        assert params["opt.ram"] == 512

    def test_multiple_inputs_raises(self, tmp_path):
        src1 = _geographic_raster(tmp_path, "s1.tif")
        src2 = _geographic_raster(tmp_path, "s2.tif")
        with pytest.raises(ValueError, match="ParallelBatch"):
            SARCalibrationStep().build_otb_params([src1, src2], tmp_path / "out.tif")


# ---------------------------------------------------------------------------
# SARCalibrationStep — execute() with mocked OTB
# ---------------------------------------------------------------------------

_FAKE_SAR_CLI = "/usr/bin/otbcli_SARCalibration"


class TestSARCalibrationExecute:
    def test_execute_calls_sar_calibration(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_SAR_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            SARCalibrationStep().execute([src], tmp_path / "out")
        assert mock_run.call_args[0][0][0] == "otbcli_SARCalibration"

    def test_compress_default_appends_gdal_suffix(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_SAR_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            SARCalibrationStep().execute([src], tmp_path / "out")
        cmd = mock_run.call_args[0][0]
        out_val = cmd[cmd.index("-out") + 1]
        assert "gdal:co:COMPRESS=DEFLATE" in out_val

    def test_compress_false_plain_path(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_SAR_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            SARCalibrationStep().execute([src], tmp_path / "out", compress=False)
        cmd = mock_run.call_args[0][0]
        out_val = cmd[cmd.index("-out") + 1]
        assert out_val.endswith("sar_calibrate_src.tif")

    def test_execute_returns_step_output(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_SAR_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()):
            result = SARCalibrationStep().execute([src], tmp_path / "out")
        assert isinstance(result, StepOutput)
        assert isinstance(result.outputs[0], FlushedOutput)

    def test_execute_output_path_prefixed(self, tmp_path):
        src = _geographic_raster(tmp_path, "src.tif")
        out_dir = tmp_path / "out"
        with patch(f"{_OTB_APP}.shutil.which", return_value=_FAKE_SAR_CLI), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()):
            result = SARCalibrationStep().execute([src], out_dir)
        assert result.outputs[0].path == out_dir / "sar_calibrate_src.tif"


# ---------------------------------------------------------------------------
# _detect_s1_borders helper
# ---------------------------------------------------------------------------


def _write_s1_like_raster(
    path: Path,
    data: np.ndarray,
    crs: CRS = None,
) -> Path:
    """Write a synthetic raster with given 2-D uint16 data."""
    if crs is None:
        crs = CRS.from_epsg(4326)
    h, w = data.shape
    transform = from_bounds(0.0, 0.0, w * 0.01, h * 0.01, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        path, "w",
        driver="GTiff", dtype="uint16", width=w, height=h,
        count=1, crs=crs, transform=transform,
    ) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


class TestDetectS1Borders:
    def test_no_borders_returns_zeros(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        path = _write_s1_like_raster(tmp_path / "clean.tif", data)
        tx, ty_s, ty_e = _detect_s1_borders(path)
        assert tx == 0
        assert ty_s == 0
        assert ty_e == 0

    def test_range_margin_detected(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        data[:, -10:] = 0  # 10-pixel right margin
        path = _write_s1_like_raster(tmp_path / "range.tif", data)
        tx, ty_s, ty_e = _detect_s1_borders(path)
        assert tx == 10

    def test_azimuth_start_margin_detected(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        data[0, :] = 0  # first row all zeros
        path = _write_s1_like_raster(tmp_path / "az_start.tif", data)
        _, ty_s, _ = _detect_s1_borders(path)
        assert ty_s == 1

    def test_azimuth_end_margin_detected(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        data[-1, :] = 0  # last row all zeros
        path = _write_s1_like_raster(tmp_path / "az_end.tif", data)
        _, _, ty_e = _detect_s1_borders(path)
        assert ty_e == 1

    def test_partial_first_row_not_detected_as_y_margin(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        data[0, :32] = 0  # only half row zero — not a full azimuth margin
        path = _write_s1_like_raster(tmp_path / "partial.tif", data)
        _, ty_s, _ = _detect_s1_borders(path)
        assert ty_s == 0


# ---------------------------------------------------------------------------
# SARBorderCutStep — class contract
# ---------------------------------------------------------------------------


class TestSARBorderCutStepContract:
    def test_registered_in_step_registry(self):
        assert StepRegistry._registry.get("sar_cut_borders") is SARBorderCutStep

    def test_class_vars(self):
        assert SARBorderCutStep.name == "sar_cut_borders"
        assert SARBorderCutStep.otb_app == "ResetMargin"


# ---------------------------------------------------------------------------
# SARBorderCutStep — execute: passthrough when no margins
# ---------------------------------------------------------------------------


class TestSARBorderCutExecute:
    def test_passthrough_when_no_margins(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        src = _write_s1_like_raster(tmp_path / "clean.tif", data)
        result = SARBorderCutStep().execute([src], tmp_path / "out")
        assert isinstance(result.outputs[0], FlushedOutput)
        assert result.outputs[0].path == src  # unchanged

    def test_passthrough_with_explicit_zero_thresholds(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        src = _write_s1_like_raster(tmp_path / "clean.tif", data)
        result = SARBorderCutStep().execute(
            [src], tmp_path / "out",
            threshold_x=0, threshold_y_start=0, threshold_y_end=0,
        )
        assert result.outputs[0].path == src

    def test_calls_otb_when_margins_detected(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        data[:, -10:] = 0
        src = _write_s1_like_raster(tmp_path / "bordered.tif", data)

        with patch(f"{_OTB_APP}.shutil.which", return_value="/usr/bin/otbcli_ResetMargin"), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            SARBorderCutStep().execute([src], tmp_path / "out")

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "otbcli_ResetMargin"

    def test_explicit_thresholds_bypass_autodetect(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16) * 100
        src = _write_s1_like_raster(tmp_path / "clean.tif", data)

        with patch(f"{_OTB_APP}.shutil.which", return_value="/usr/bin/otbcli_ResetMargin"), \
             patch(f"{_OTB_APP}.subprocess.run", return_value=_mock_proc()) as mock_run:
            SARBorderCutStep().execute(
                [src], tmp_path / "out",
                threshold_x=20, threshold_y_start=0, threshold_y_end=0,
            )

        cmd = mock_run.call_args[0][0]
        tx_idx = cmd.index("-threshold.x")
        assert cmd[tx_idx + 1] == "20"

    def test_multiple_inputs_raises(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16)
        s1 = _write_s1_like_raster(tmp_path / "s1.tif", data)
        s2 = _write_s1_like_raster(tmp_path / "s2.tif", data)
        with pytest.raises(ValueError, match="ParallelBatch"):
            SARBorderCutStep().execute([s1, s2], tmp_path / "out")

    def test_build_otb_params_sets_threshold_keys(self, tmp_path):
        data = np.ones((64, 64), dtype=np.uint16)
        src = _write_s1_like_raster(tmp_path / "src.tif", data)
        params = SARBorderCutStep().build_otb_params(
            [src], tmp_path / "out.tif",
            threshold_x=15, threshold_y_start=2, threshold_y_end=3,
        )
        assert params["threshold.x"] == 15
        assert params["threshold.y.start"] == 2
        assert params["threshold.y.end"] == 3
        assert "out" not in params
