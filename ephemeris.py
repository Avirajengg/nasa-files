"""
Sun and Earth azimuth/elevation as seen from a point on the Moon.

Uses NASA NAIF SPICE (via SpiceyPy). Needed kernels in the data folder:
  naif0012.tls                  leap seconds
  de440s.bsp                    planetary ephemeris (DE440)
  moon_pa_de440_200625.bpc      lunar orientation (binary PCK)
  moon_de440_*.tf               lunar frame kernel (defines MOON_ME_DE440_ME421)

Frame used: MOON_ME_DE440_ME421, the DE440 realization of the Mean Earth /
polar axis frame, built to align with the DE421 ME frame. This is the frame
the PGDA terrain products are registered in.
"""
import glob
import os

import numpy as np
import spiceypy as spice

MOON_FRAME = "MOON_ME_DE440_ME421"
MOON_RADIUS_KM = 1737.4          # lunar reference radius
ABCORR = "LT+S"                  # light-time + stellar aberration
BODY_RADIUS_KM = {"SUN": 695_700.0, "EARTH": 6_371.0}   # mean radii

_KERNEL_PATTERNS = [
    "naif0012.tls",
    "de440s.bsp",
    "moon_pa_de440_200625.bpc",
    "moon_de440_*.tf",
]
_loaded = False


def _candidate_dirs(data_dir):
    """Places to look for the kernels, in order."""
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = []
    if data_dir:
        dirs.append(data_dir)
    dirs += [
        os.path.join(os.getcwd(), "data"),
        os.path.join(here, "..", "data"),
        os.path.join(here, "data"),
    ]
    return [os.path.normpath(d) for d in dirs]


def _find_kernel(folder, pattern):
    """Newest file matching pattern. Also accepts a hidden '.txt' added by Windows."""
    for pat in (pattern, pattern + ".txt"):
        matches = sorted(glob.glob(os.path.join(folder, pat)))
        if matches:
            return matches[-1]
    return None


def resolve_kernels(data_dir=None):
    """Return the list of kernel paths, or raise an error that says what is wrong."""
    dirs = _candidate_dirs(data_dir)
    folder = next((d for d in dirs if _find_kernel(d, "de440s.bsp")), None)
    if folder is None:
        existing = [d for d in dirs if os.path.isdir(d)]
        listing = "\n".join(
            f"  {d}: {sorted(os.listdir(d))}" for d in existing
        ) or "  (none of these folders exist)"
        raise FileNotFoundError(
            "Could not find the kernel folder (looked for de440s.bsp in):\n  "
            + "\n  ".join(dirs)
            + "\nFolders that do exist and what is in them:\n" + listing
        )
    paths, missing = [], []
    for pattern in _KERNEL_PATTERNS:
        found = _find_kernel(folder, pattern)
        (paths if found else missing).append(found or pattern)
    if missing:
        raise FileNotFoundError(
            f"In {folder} these kernels are missing: {missing}\n"
            f"The folder contains: {sorted(os.listdir(folder))}\n"
            "Check spelling, and that Windows has not added a hidden '.txt' to the name."
        )
    return paths


def load_kernels(data_dir=None, verbose=True):
    """Load the SPICE kernels once. Finds the data folder automatically."""
    global _loaded
    if _loaded:
        return
    for path in resolve_kernels(data_dir):
        spice.furnsh(path)
        if verbose:
            print("loaded:", path)
    if spice.namfrm(MOON_FRAME) == 0:
        raise RuntimeError(
            f"Frame {MOON_FRAME} is not defined. Check the moon_de440_*.tf file "
            "(it should start with 'KPL/FK')."
        )
    _loaded = True


def utc_to_et(utc_string):
    """UTC string such as '2026-11-01T00:00:00' to SPICE ephemeris time (s)."""
    return spice.str2et(utc_string)


def make_times(start_utc, days, step_hours=1.0):
    """Array of ephemeris times from start_utc, every step_hours, for `days`."""
    et0 = utc_to_et(start_utc)
    n = int(round(days * 24.0 / step_hours))
    return et0 + np.arange(n) * step_hours * 3600.0


def local_frame(lat_deg, lon_deg):
    """
    East, North, Up unit vectors at (lat, lon) in the Moon-fixed frame.
    Written in closed form so it also works exactly at the pole.
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return east, north, up


def site_vector(lat_deg, lon_deg, height_m=0.0):
    """Site position (km) from the Moon's center, in the Moon-fixed frame."""
    _, _, up = local_frame(lat_deg, lon_deg)
    return (MOON_RADIUS_KM + height_m / 1000.0) * up


def vector_to_azel(v, east, north, up):
    """
    Convert direction vectors (N x 3, or one 3-vector) to azimuth/elevation.
    Azimuth: degrees clockwise from north, 0-360. Elevation: degrees above horizon.
    """
    v = np.atleast_2d(v)
    norm = np.linalg.norm(v, axis=1)
    az = np.mod(np.degrees(np.arctan2(v @ east, v @ north)) + 360.0, 360.0)
    el = np.degrees(np.arcsin(np.clip((v @ up) / norm, -1.0, 1.0)))
    return az, el


def body_azel(body, lat_deg, lon_deg, height_m, ets):
    """Azimuth and elevation (degrees) of `body` at the site for each time in ets."""
    ets = np.atleast_1d(np.asarray(ets, dtype=float))
    east, north, up = local_frame(lat_deg, lon_deg)
    site = site_vector(lat_deg, lon_deg, height_m)
    pos, _ = spice.spkpos(body, ets, MOON_FRAME, ABCORR, "MOON")
    return vector_to_azel(np.asarray(pos) - site, east, north, up)


def sun_earth_azel(lat_deg, lon_deg, height_m, ets):
    """Returns dict with keys 'sun' and 'earth', each (azimuth, elevation) arrays."""
    return {
        "sun": body_azel("SUN", lat_deg, lon_deg, height_m, ets),
        "earth": body_azel("EARTH", lat_deg, lon_deg, height_m, ets),
    }


def angular_radius_deg(radius_km, distance_km):
    """Apparent angular radius (degrees) of a sphere of radius_km at distance_km."""
    return np.degrees(np.arcsin(radius_km / np.asarray(distance_km, dtype=float)))


def body_angular_radius(body, ets):
    """Apparent angular radius (deg) of SUN or EARTH, as seen from the Moon, per time."""
    ets = np.atleast_1d(np.asarray(ets, dtype=float))
    pos, _ = spice.spkpos(body, ets, MOON_FRAME, ABCORR, "MOON")
    return angular_radius_deg(BODY_RADIUS_KM[body], np.linalg.norm(pos, axis=1))