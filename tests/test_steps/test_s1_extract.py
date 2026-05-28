"""Tests for S1ExtractStep."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from eo_pipe.steps.raster.s1_extract import S1ExtractStep, _parse_gcps

# ---------------------------------------------------------------------------
# Synthetic S1 fixture helpers
# ---------------------------------------------------------------------------

_PRODUCT_NAME = "S1C_IW_GRDH_1SDV_20260101T000000_20260101T000030_000001_000001_0001.SAFE"
_W, _H = 32, 32

_GCP_GRID = [
    (0, 0, 48.0, 2.0),
    (0, 31, 48.0, 2.32),
    (31, 0, 47.68, 2.0),
    (31, 31, 47.68, 2.32),
]  # (line, pixel, lat, lon)


def _meas_filename(pol: str) -> str:
    return f"s1c-iw-grd-{pol}-20260101t000000-20260101t000030-000001-000001-001.tiff"


def _ann_filename(pol: str) -> str:
    return f"s1c-iw-grd-{pol}-20260101t000000-20260101t000030-000001-000001-001.xml"


def _write_measurement(path: Path, pol: str) -> Path:
    """Write a minimal single-band uint16 raster with no CRS (raw S1 style)."""
    tif = path / "measurement" / _meas_filename(pol)
    tif.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(pol) % (2**31))
    data = rng.integers(0, 65535, (_H, _W), dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "height": _H,
        "width": _W,
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data, 1)
    return tif


def _write_annotation(path: Path, pol: str) -> Path:
    """Write a minimal annotation XML with a 2×2 GCP grid."""
    xml_path = path / "annotation" / _ann_filename(pol)
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("product")
    grid = ET.SubElement(ET.SubElement(root, "geolocationGrid"), "geolocationGridPointList")
    grid.set("count", str(len(_GCP_GRID)))
    for line, pixel, lat, lon in _GCP_GRID:
        pt = ET.SubElement(grid, "geolocationGridPoint")
        ET.SubElement(pt, "line").text = str(line)
        ET.SubElement(pt, "pixel").text = str(pixel)
        ET.SubElement(pt, "latitude").text = str(lat)
        ET.SubElement(pt, "longitude").text = str(lon)
        ET.SubElement(pt, "height").text = "0.0"

    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def _make_safe_dir(base: Path, polarisations: list[str] = None) -> Path:
    """Build a minimal SAFE directory with given polarisations."""
    if polarisations is None:
        polarisations = ["vh", "vv"]
    safe = base / _PRODUCT_NAME
    safe.mkdir(parents=True, exist_ok=True)
    for pol in polarisations:
        _write_measurement(safe, pol)
        _write_annotation(safe, pol)
    return safe


def _make_zip(safe_dir: Path, zip_dir: Path) -> Path:
    """Zip a SAFE directory and return the zip path."""
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / (safe_dir.name.replace(".SAFE", "") + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in safe_dir.rglob("*"):
            zf.write(file, file.relative_to(safe_dir.parent))
    return zip_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def safe_dir(tmp_path: Path) -> Path:
    return _make_safe_dir(tmp_path / "safe")


@pytest.fixture()
def safe_zip(tmp_path: Path) -> Path:
    safe = _make_safe_dir(tmp_path / "safe")
    return _make_zip(safe, tmp_path / "zips")


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# _parse_gcps unit tests
# ---------------------------------------------------------------------------


class TestParseGcps:
    def test_count(self, tmp_path):
        ann = _write_annotation(tmp_path, "vh")
        gcps = _parse_gcps(ann)
        assert len(gcps) == len(_GCP_GRID)

    def test_coordinates(self, tmp_path):
        ann = _write_annotation(tmp_path, "vh")
        gcps = _parse_gcps(ann)
        first = gcps[0]
        assert first.row == _GCP_GRID[0][0]
        assert first.col == _GCP_GRID[0][1]
        assert first.y == pytest.approx(_GCP_GRID[0][2])
        assert first.x == pytest.approx(_GCP_GRID[0][3])


# ---------------------------------------------------------------------------
# S1ExtractStep — SAFE directory input
# ---------------------------------------------------------------------------


class TestS1ExtractStepSafeDir:
    def test_step_name(self):
        assert S1ExtractStep.name == "s1_extract"

    def test_single_pol_output_exists(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH"]).flush_all()

        assert len(result.outputs) == 1
        assert result.outputs[0].exists()

    def test_output_path_structure(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH"]).flush_all()

        out = result.outputs[0]
        # output_dir / {product_stem} / measurement / {original_filename}
        assert out.parent.parent.name == _PRODUCT_NAME.replace(".SAFE", "")
        assert out.parent.name == "measurement"
        assert out.name == _meas_filename("vh")

    def test_two_pols_both_extracted(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH", "VV"]).flush_all()

        assert len(result.outputs) == 2
        names = {p.name for p in result.outputs}
        assert names == {_meas_filename("vh"), _meas_filename("vv")}

    def test_gcps_embedded(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH"]).flush_all()

        with rasterio.open(result.outputs[0]) as ds:
            gcps, crs = ds.gcps
        assert len(gcps) == len(_GCP_GRID)
        assert crs.to_epsg() == 4326

    def test_no_affine_transform_in_output(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH"]).flush_all()

        with rasterio.open(result.outputs[0]) as ds:
            assert ds.transform.is_identity

    def test_pixel_values_preserved(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH"]).flush_all()

        meas = safe_dir / "measurement" / _meas_filename("vh")
        with rasterio.open(meas) as src:
            src_data = src.read(1)
        with rasterio.open(result.outputs[0]) as dst:
            dst_data = dst.read(1)

        np.testing.assert_array_equal(src_data, dst_data)

    def test_missing_pol_skipped(self, safe_dir, output_dir):
        """HH not in fixture — should be skipped, not raise."""
        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["HH"]).flush_all()
        assert len(result.outputs) == 0

    def test_invalid_pol_raises(self, safe_dir, output_dir):
        step = S1ExtractStep()
        with pytest.raises(ValueError, match="Unknown polarisation"):
            step.execute([safe_dir], output_dir, polarisations=["XX"])

    def test_missing_annotation_no_gcps(self, safe_dir, output_dir):
        """Remove annotation XML — output still written, but no GCPs."""
        (safe_dir / "annotation" / _ann_filename("vh")).unlink()

        step = S1ExtractStep()
        result = step.execute([safe_dir], output_dir, polarisations=["VH"]).flush_all()

        assert result.outputs[0].exists()
        with rasterio.open(result.outputs[0]) as ds:
            gcps, _ = ds.gcps
        assert gcps == []


# ---------------------------------------------------------------------------
# S1ExtractStep — zip input
# ---------------------------------------------------------------------------


class TestS1ExtractStepZip:
    def test_extracts_from_zip(self, safe_zip, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_zip], output_dir, polarisations=["VH"]).flush_all()

        assert len(result.outputs) == 1
        assert result.outputs[0].exists()

    def test_zip_output_path_structure(self, safe_zip, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_zip], output_dir, polarisations=["VH"]).flush_all()

        out = result.outputs[0]
        assert out.parent.parent.name == _PRODUCT_NAME.replace(".SAFE", "")
        assert out.parent.name == "measurement"
        assert out.name == _meas_filename("vh")

    def test_zip_gcps_embedded(self, safe_zip, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_zip], output_dir, polarisations=["VH"]).flush_all()

        with rasterio.open(result.outputs[0]) as ds:
            gcps, crs = ds.gcps
        assert len(gcps) == len(_GCP_GRID)
        assert crs.to_epsg() == 4326

    def test_zip_two_pols(self, safe_zip, output_dir):
        step = S1ExtractStep()
        result = step.execute([safe_zip], output_dir, polarisations=["VH", "VV"]).flush_all()

        assert len(result.outputs) == 2

    def test_invalid_input_raises(self, tmp_path, output_dir):
        bad = tmp_path / "not_a_safe.tif"
        bad.touch()
        step = S1ExtractStep()
        with pytest.raises(ValueError, match="Input must be"):
            step.execute([bad], output_dir, polarisations=["VH"])


# ---------------------------------------------------------------------------
# S1ExtractStep — fast copy (extract_safe=False)
# ---------------------------------------------------------------------------


class TestS1ExtractFastCopy:
    def test_fast_copy_output_exists(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute(
            [safe_dir], output_dir, polarisations=["VH"], extract_safe=False
        ).flush_all()

        assert len(result.outputs) == 1
        assert result.outputs[0].exists()

    def test_fast_copy_no_gcps(self, safe_dir, output_dir):
        """Fast copy skips annotation parsing — no GCPs embedded."""
        step = S1ExtractStep()
        result = step.execute(
            [safe_dir], output_dir, polarisations=["VH"], extract_safe=False
        ).flush_all()

        with rasterio.open(result.outputs[0]) as ds:
            gcps, _ = ds.gcps
        assert gcps == []

    def test_fast_copy_pixel_values_preserved(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute(
            [safe_dir], output_dir, polarisations=["VH"], extract_safe=False
        ).flush_all()

        meas = safe_dir / "measurement" / _meas_filename("vh")
        with rasterio.open(meas) as src:
            src_data = src.read(1)
        with rasterio.open(result.outputs[0]) as dst:
            dst_data = dst.read(1)

        np.testing.assert_array_equal(src_data, dst_data)

    def test_fast_copy_two_pols(self, safe_dir, output_dir):
        step = S1ExtractStep()
        result = step.execute(
            [safe_dir], output_dir, polarisations=["VH", "VV"], extract_safe=False
        ).flush_all()

        assert len(result.outputs) == 2
