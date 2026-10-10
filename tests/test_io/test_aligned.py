"""Round-trip tests of the Aligned raster tag and filename contract."""

import pytest

from eo_pipe.io.aligned import (
    TAG_POLARIZATION,
    aligned_name,
    identify,
    polarization_of,
    tags_for,
)


def _written(stem="S1A_x_vv", pol=None, count=1, descriptions=(None,)):
    pol = polarization_of(count, descriptions, stem, pol)
    return tags_for({"K": "v"}, tile="31TCJ", token="20250314T051233", source=stem,
                    params={"resolution": 10.0}, polarization=pol)


class TestPolarizationOf:
    def test_override_wins(self):
        assert polarization_of(1, ("VH",), "x_vv", "hh") == "HH"

    def test_description_before_stem(self):
        assert polarization_of(1, ("vh",), "x_vv") == "VH"

    def test_stem_fallback(self):
        assert polarization_of(1, (None,), "x_vv") == "VV"

    def test_multiband_has_none(self):
        assert polarization_of(2, ("VV", "VH"), "x_vv") is None


class TestRoundTrip:
    def test_identity_survives_write_then_read(self):
        tags = _written()
        ident = identify("renamed.tif", tags, 1, (None,))
        assert (ident.tile, ident.iso, ident.variables) == ("31TCJ", "2025-03-14T05:12:33", ["VV"])
        assert ident.tags["K"] == "v"

    def test_tags_win_over_filename(self):
        tags = _written()
        name = aligned_name("30TXN", "20200101", "other_vh")
        ident = identify(name, tags, 1, (None,))
        assert ident.tile == "31TCJ" and ident.variables == ["VV"]

    def test_multiband_uses_descriptions(self):
        tags = _written(count=2, descriptions=("a", None))
        assert TAG_POLARIZATION not in tags
        assert identify("x.tif", tags, 2, ("a", None)).variables == ["a", "band_2"]


class TestFilenameFallback:
    def test_untagged_file_reads_name_and_stem_polarization(self):
        ident = identify(aligned_name("31TCJ", "20250314", "S1A_vh"), {}, 1, (None,))
        assert (ident.tile, ident.iso, ident.variables) == ("31TCJ", "2025-03-14", ["VH"])

    def test_untagged_without_pol_uses_band_names(self):
        ident = identify(aligned_name("31TCJ", "20250314", "scene"), {}, 1, (None,))
        assert ident.variables == ["band_1"]

    def test_unparseable_untagged_raises(self):
        with pytest.raises(ValueError, match="TILE/DATETIME"):
            identify("random.tif", {}, 1, (None,))
