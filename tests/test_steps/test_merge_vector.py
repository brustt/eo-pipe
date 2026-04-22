"""Tests for MergeVectorStep."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from eo_pipe.io.output_types import GpkgFormat, ParquetFormat, ShapefileFormat
from eo_pipe.steps.vector.merge import MergeVectorStep


def _write_shp(path: Path, geoms, crs="EPSG:4326") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"geometry": geoms, "val": range(len(geoms))}, crs=crs)
    gdf.to_file(path)
    return path


class TestMergeVectorStep:
    def test_merge_two_shapefiles_default_gpkg(self, tmp_path, output_dir):
        """Default format is GPKG."""
        a = _write_shp(tmp_path / "a.shp", [box(0, 0, 1, 1)])
        b = _write_shp(tmp_path / "b.shp", [box(1, 0, 2, 1)])

        step = MergeVectorStep()
        result = step.execute([a, b], output_dir).flush_all()

        assert len(result.outputs) == 1
        assert result.outputs[0].suffix == ".gpkg"
        assert result.outputs[0].exists()

        merged = gpd.read_file(result.outputs[0])
        assert len(merged) == 2

    def test_output_name_param(self, tmp_path, output_dir):
        a = _write_shp(tmp_path / "a.shp", [box(0, 0, 1, 1)])
        b = _write_shp(tmp_path / "b.shp", [box(1, 0, 2, 1)])

        step = MergeVectorStep()
        result = step.execute([a, b], output_dir, output_name="combined").flush_all()
        assert result.outputs[0].stem == "combined"

    def test_shapefile_format(self, tmp_path, output_dir):
        a = _write_shp(tmp_path / "a.shp", [box(0, 0, 1, 1)])
        b = _write_shp(tmp_path / "b.shp", [box(1, 0, 2, 1)])

        step = MergeVectorStep()
        result = step.execute(
            [a, b], output_dir, output_name="out", fmt=ShapefileFormat()
        ).flush_all()
        assert result.outputs[0].suffix == ".shp"
        assert result.outputs[0].exists()

    def test_parquet_format(self, tmp_path, output_dir):
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        a = _write_shp(tmp_path / "a.shp", [box(0, 0, 1, 1)])
        b = _write_shp(tmp_path / "b.shp", [box(1, 0, 2, 1)])

        step = MergeVectorStep()
        result = step.execute(
            [a, b], output_dir, output_name="out", fmt=ParquetFormat()
        ).flush_all()
        assert result.outputs[0].suffix == ".parquet"
        assert result.outputs[0].exists()

        gdf = gpd.read_parquet(result.outputs[0])
        assert len(gdf) == 2

    def test_dissolve_produces_one_geometry(self, tmp_path, output_dir):
        a = _write_shp(tmp_path / "a.shp", [box(0, 0, 1, 1)])
        b = _write_shp(tmp_path / "b.shp", [box(0.5, 0, 1.5, 1)])

        step = MergeVectorStep()
        result = step.execute([a, b], output_dir, dissolve=True).flush_all()

        merged = gpd.read_file(result.outputs[0])
        assert len(merged) == 1

    def test_missing_file_raises(self, tmp_path, output_dir):
        a = _write_shp(tmp_path / "a.shp", [box(0, 0, 1, 1)])
        missing = tmp_path / "does_not_exist.shp"

        step = MergeVectorStep()
        with pytest.raises(FileNotFoundError):
            step.execute([a, missing], output_dir)

    def test_empty_inputs_raises(self, output_dir):
        step = MergeVectorStep()
        with pytest.raises((ValueError, FileNotFoundError)):
            step.execute([], output_dir)

    def test_step_name(self):
        assert MergeVectorStep.name == "merge_vector"
