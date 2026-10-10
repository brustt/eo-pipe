import pytest
from affine import Affine

from eo_pipe.io.lattice import Lattice, intersect

LAT = Lattice(origin_x=500_000.0, origin_y=4_800_000.0, res_x=10.0)


def test_square_pixels_by_default():
    assert LAT.res_y == 10.0


def test_snap_outward_expands_to_whole_pixels():
    minx, miny, maxx, maxy = LAT.snap_outward((500_003.0, 4_799_903.0, 500_097.0, 4_799_997.0))
    assert (minx, miny, maxx, maxy) == (500_000.0, 4_799_900.0, 500_100.0, 4_800_000.0)


def test_snap_outward_keeps_on_lattice_bounds():
    b = (500_010.0, 4_799_900.0, 500_100.0, 4_799_990.0)
    assert LAT.snap_outward(b) == b


def test_snap_outward_absorbs_float_noise():
    b = (500_010.0 + 1e-9, 4_799_900.0, 500_100.0 - 1e-9, 4_799_990.0)
    assert LAT.snap_outward(b) == (500_010.0, 4_799_900.0, 500_100.0, 4_799_990.0)


def test_snap_outward_non_square_pixels():
    lat = Lattice(0.0, 0.0, 10.0, 20.0)
    assert lat.snap_outward((1.0, -39.0, 11.0, -1.0)) == (0.0, -40.0, 20.0, 0.0)


def test_shape():
    assert LAT.shape((500_000.0, 4_799_900.0, 500_100.0, 4_800_000.0)) == (10, 10)


def test_holds():
    assert LAT.holds(500_030.0, 4_799_960.0)
    assert not LAT.holds(500_035.0, 4_799_960.0)
    assert not LAT.holds(500_030.0, 4_799_955.0)


def test_holds_transform():
    assert LAT.holds_transform(Affine(10, 0, 500_020, 0, -10, 4_799_980))
    assert not LAT.holds_transform(Affine(20, 0, 500_020, 0, -20, 4_799_980))
    assert not LAT.holds_transform(Affine(10, 0, 500_025, 0, -10, 4_799_980))
    assert not LAT.holds_transform(Affine(10, 1, 500_020, 0, -10, 4_799_980))


def test_offset_and_anchor():
    assert LAT.offset(500_030.0, 4_799_950.0) == (3, 5)
    sub = LAT.anchored_at((500_020.0, 4_799_900.0, 500_100.0, 4_799_990.0))
    assert sub.offset(500_050.0, 4_799_970.0) == (3, 2)


def test_intersect():
    assert intersect((0, 0, 10, 10), (5, 5, 20, 20)) == (5, 5, 10, 10)
    assert intersect((0, 0, 10, 10), (10, 0, 20, 10)) is None
    assert intersect((0, 0, 10, 10), (20, 20, 30, 30)) is None
