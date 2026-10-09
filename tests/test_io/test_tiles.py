"""Tests for Sentinel-2 Tile lookup."""

from pathlib import Path

import pytest
from shapely.geometry import box

from eo_pipe.io.tiles import S2_TILE_GRID_PATH, find_tiles, tile_epsg


class TestTileEpsg:
    @pytest.mark.parametrize(
        "name, epsg",
        [("31TCJ", 32631), ("32TLR", 32632), ("01GDM", 32701), ("55HFA", 32755), ("30UWV", 32630)],
    )
    def test_utm_zone_and_hemisphere_from_name(self, name, epsg):
        assert tile_epsg(name) == epsg


class TestFindTiles:
    def test_returns_tile_with_real_sentinel2_origin(self, tile_grid):
        # Footprint well inside 31TCJ (lon ~1.2, lat ~43.9)
        footprint = box(1.2, 43.9, 1.21, 43.91)
        tiles = find_tiles(footprint, tile_grid)

        names = {t.name for t in tiles}
        assert "31TCJ" in names
        tile = next(t for t in tiles if t.name == "31TCJ")
        assert tile.crs.to_epsg() == 32631
        # Known Sentinel-2 granule origin of 31TCJ
        assert (tile.ul_x, tile.ul_y) == (300000.0, 4900020.0)
        assert tile.bounds == (300000.0, 4790220.0, 409800.0, 4900020.0)

    def test_neighbouring_tile_origin_has_s2_offset(self, tile_grid):
        footprint = box(2.6, 43.9, 2.61, 43.91)
        tile = next(t for t in find_tiles(footprint, tile_grid) if t.name == "31TDJ")
        assert (tile.ul_x, tile.ul_y) == (399960.0, 4900020.0)

    def test_footprint_outside_grid_returns_nothing(self, tile_grid):
        assert find_tiles(box(100, -40, 100.1, -39.9), tile_grid) == []

    def test_footprint_in_overlap_returns_both_tiles(self, tile_grid):
        # 31TCJ spans x to 409800, 31TDJ starts at 399960 -> overlap near lon 1.8
        footprint = box(1.8, 43.9, 1.81, 43.91)
        names = {t.name for t in find_tiles(footprint, tile_grid)}
        assert names >= {"31TCJ", "31TDJ"}


@pytest.mark.skipif(not S2_TILE_GRID_PATH.exists(), reason="real Sentinel-2 grid not available")
class TestRealGrid:
    def test_real_origins_match_known_granules(self):
        tiles = {t.name: t for t in find_tiles(box(1.2, 43.9, 2.61, 43.91), S2_TILE_GRID_PATH)}
        assert (tiles["31TCJ"].ul_x, tiles["31TCJ"].ul_y) == (300000.0, 4900020.0)
        assert (tiles["31TDJ"].ul_x, tiles["31TDJ"].ul_y) == (399960.0, 4900020.0)
