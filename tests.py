"""Checks of the local frame and az/el conversion. No SPICE kernels needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ephemeris as eph


def test_frame_is_orthonormal_everywhere():
    for lat in (-90, -89.9, -45, 0, 60):
        for lon in (0, 129, 270):
            e, n, u = eph.local_frame(lat, lon)
            M = np.array([e, n, u])
            assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
            assert np.allclose(np.cross(e, n), u, atol=1e-12)   # right-handed


def test_azel_directions():
    e, n, u = eph.local_frame(-60, 30)
    az, el = eph.vector_to_azel(np.array([n, e, -n, -e, u]), e, n, u)
    assert np.allclose(el, [0, 0, 0, 0, 90])
    assert np.allclose(az[:4], [0, 90, 180, 270])


def test_pole_site_matches_radius():
    assert np.isclose(np.linalg.norm(eph.site_vector(-90, 0, 0)), eph.MOON_RADIUS_KM)
    assert np.isclose(np.linalg.norm(eph.site_vector(-90, 0, 1000)), eph.MOON_RADIUS_KM + 1.0)