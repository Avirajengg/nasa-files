"""
Terrain horizon mask from a PGDA elevation tile (polar stereographic, meters).

For each compass azimuth we walk outward from the site across the tile and
record the highest elevation angle the terrain reaches. The Sun (or Earth) is
visible only when its elevation exceeds that angle in its azimuth.

Limits (reported, not hidden):
  * The horizon is only as far as the tile. `range_m` says how far each ray
    could look before it left the tile; terrain beyond that is not seen.
  * Lunar curvature is included, so distant terrain drops below the local plane.
  * The Sun is treated as a point (no disk).
"""
import numpy as np

MOON_RADIUS_M = 1_737_400.0


def local_directions(lon_deg):
    """
    Unit vectors (in map x, y) for local north and east at east-longitude lon.
    South polar stereographic: x = rho*sin(lon), y = rho*cos(lon), and local
    north points away from the pole (increasing rho).
    """
    lon = np.radians(lon_deg)
    north = np.array([np.sin(lon), np.cos(lon)])
    east = np.array([np.cos(lon), -np.sin(lon)])
    return north, east


def compute_horizon(z, transform, x0, y0, lon_deg, site_height_m,
                    n_az=360, step_m=5.0, max_range_m=None, nodata=None):
    """
    z              2-D elevation array (m above the 1737.4 km sphere)
    transform      affine transform of the tile (north-up, no rotation)
    x0, y0         site position in projected meters
    lon_deg        site east longitude (to orient north/east)
    site_height_m  absolute height of the observer (ground elevation + mast)
    Returns dict: az_deg, horizon_deg, range_m (arrays of length n_az).
    """
    res = transform.a
    left, top = transform.c, transform.f
    nrow, ncol = z.shape
    if max_range_m is None:
        max_range_m = float(np.hypot(nrow, ncol) * res)

    d = np.arange(step_m, max_range_m, step_m)                 # distances (m)
    az = np.arange(n_az) * 360.0 / n_az
    north, east = local_directions(lon_deg)
    a = np.radians(az)[:, None]
    dir_x = np.cos(a) * north[0] + np.sin(a) * east[0]
    dir_y = np.cos(a) * north[1] + np.sin(a) * east[1]

    x = x0 + dir_x * d[None, :]
    y = y0 + dir_y * d[None, :]
    col = np.floor((x - left) / res).astype(int)
    row = np.floor((top - y) / res).astype(int)
    inside = (row >= 0) & (row < nrow) & (col >= 0) & (col < ncol)
    rr = np.clip(row, 0, nrow - 1)
    cc = np.clip(col, 0, ncol - 1)
    h = z[rr, cc].astype(float)
    valid = inside.copy()
    if nodata is not None:
        valid &= h != nodata
    valid &= np.isfinite(h)

    # exact spherical geometry: terrain at central angle theta, radius R+h
    theta = d[None, :] / MOON_RADIUS_M
    R = MOON_RADIUS_M
    vert = (R + h) * np.cos(theta) - (R + site_height_m)
    horiz = (R + h) * np.sin(theta)
    ang = np.degrees(np.arctan2(vert, horiz))
    ang = np.where(valid, ang, -np.inf)

    horizon = ang.max(axis=1)
    # distance each ray could look before leaving the tile / hitting no-data
    first_bad = np.where(valid.all(axis=1), valid.shape[1], np.argmin(valid, axis=1))
    range_m = np.where(first_bad > 0, d[np.clip(first_bad - 1, 0, len(d) - 1)], 0.0)
    horizon = np.where(np.isfinite(horizon), horizon, np.nan)
    return {"az_deg": az, "horizon_deg": horizon, "range_m": range_m}


def horizon_for_site(ds, site, observer_m=2.0, **kw):
    """Convenience wrapper: ds is an open rasterio dataset, site from sites.py."""
    z = ds.read(1)
    return compute_horizon(z, ds.transform, site["x"], site["y"], site["lon"],
                           site["elev_m"] + observer_m, nodata=ds.nodata, **kw)


def horizon_at(az_query_deg, horizon):
    """Horizon elevation (deg) at arbitrary azimuths, interpolated around 360."""
    return np.interp(np.asarray(az_query_deg) % 360.0, horizon["az_deg"],
                     horizon["horizon_deg"], period=360.0)


def visible(az_deg, el_deg, horizon, radius_deg=0.0, mode="center"):
    """
    True where the body is above the terrain horizon.
      mode="center": the body's centre is above the horizon (point source)
      mode="any":    any part of the disk is above it (upper limb: el + r > horizon)
      mode="full":   the whole disk is above it (lower limb: el - r > horizon)
    radius_deg may be a scalar or an array matching el_deg. The horizon is read
    at the disk centre's azimuth (a small approximation across the disk width).
    """
    h = horizon_at(az_deg, horizon)
    el = np.asarray(el_deg)
    if mode == "center":
        return el > h
    if mode == "any":
        return el + radius_deg > h
    if mode == "full":
        return el - radius_deg > h
    raise ValueError("mode must be 'center', 'any' or 'full'")