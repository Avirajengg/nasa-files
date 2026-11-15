"""Horizon and metrics tests on synthetic terrain (numpy + affine only)."""
import os
import sys

import numpy as np
from affine import Affine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import horizon as hz
import metrics as mt

RES = 5.0


def tile(extent=3000):
    n = int(2 * extent / RES)
    transform = Affine(RES, 0, -extent, 0, -RES, extent)   # left=-extent, top=+extent
    return np.zeros((n, n), dtype="float32"), transform


def row_col(T, x, y):
    return int((T.f - y) / RES), int((x - T.c) / RES)


def test_wall_east_and_north_at_lon0():
    # lon 0: north = +y, east = +x
    z, T = tile()
    c0 = row_col(T, 500, 0)[1]
    z[:, c0:c0 + 4] = 50.0                           # wall 500 m east, 50 m high
    r0 = row_col(T, 0, 1500)[0]
    z[r0:r0 + 4, :] = 30.0                           # wall 500 m north, 30 m high
    h = hz.compute_horizon(z, T, 0.0, 1000.0, 0.0, 0.0)
    assert abs(h["horizon_deg"][90] - np.degrees(np.arctan(50 / 500))) < 0.4
    assert abs(h["horizon_deg"][0] - np.degrees(np.arctan(30 / 500))) < 0.4
    assert h["horizon_deg"][180] < 0.1 and h["horizon_deg"][270] < 0.1


def test_orientation_follows_longitude():
    # lon 90: north = +x, east = -y
    z, T = tile()
    c0 = row_col(T, 1500, 0)[1]
    z[:, c0:c0 + 4] = 40.0                           # 500 m toward +x = north
    r0 = row_col(T, 0, -500)[0]
    z[r0:r0 + 4, :] = 20.0                           # 500 m toward -y = east
    h = hz.compute_horizon(z, T, 1000.0, 0.0, 90.0, 0.0)
    assert abs(h["horizon_deg"][0] - np.degrees(np.arctan(40 / 500))) < 0.4
    assert abs(h["horizon_deg"][90] - np.degrees(np.arctan(20 / 500))) < 0.4


def test_curvature_keeps_flat_ground_at_or_below_zero():
    z, T = tile()
    h = hz.compute_horizon(z, T, 0.0, 1000.0, 0.0, 0.0)
    assert np.all(h["horizon_deg"] <= 0.0)


def test_range_reports_tile_exit():
    z, T = tile(extent=3000)
    h = hz.compute_horizon(z, T, 0.0, 1000.0, 0.0, 0.0)
    assert h["range_m"][0] < 2100 and h["range_m"][180] > 3900   # north edge near, south far


def test_visible_uses_horizon():
    hor = {"az_deg": np.arange(360.0), "horizon_deg": np.full(360, 2.0)}
    assert list(hz.visible([10, 10], [1.0, 3.0], hor)) == [False, True]


def test_metrics():
    m = np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool)
    assert mt.fraction_true(m) == 3 / 7
    assert mt.runs(m, True) == [(0, 2), (5, 1)]
    assert mt.longest_run_hours(m, 6.0, value=False) == 18.0


def test_disk_modes():
    hor = {"az_deg": np.arange(360.0), "horizon_deg": np.zeros(360)}
    el = np.array([-0.4, -0.1, 0.1, 0.4])
    r = 0.27
    assert list(hz.visible([0] * 4, el, hor, r, "center")) == [False, False, True, True]
    assert list(hz.visible([0] * 4, el, hor, r, "any")) == [False, True, True, True]
    assert list(hz.visible([0] * 4, el, hor, r, "full")) == [False, False, False, True]


def test_angular_radius():
    import ephemeris as eph
    assert abs(eph.angular_radius_deg(695700.0, 149597870.7) - 0.2666) < 0.001   # Sun at 1 AU
    assert abs(eph.angular_radius_deg(6371.0, 384400.0) - 0.9496) < 0.002        # Earth from Moon