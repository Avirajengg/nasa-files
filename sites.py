"""
Turn a pixel of a PGDA terrain tile (polar stereographic, meters) into the
latitude, longitude and elevation that ephemeris.sun_earth_azel needs.
"""
import numpy as np
import rasterio
from pyproj import CRS, Transformer


def _transformer(ds):
    crs = CRS.from_wkt(ds.crs.to_wkt())
    return Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)


def xy_to_lonlat(ds, x, y):
    """Projected meters (x easting, y northing) -> (lat_deg, lon_deg east, 0..360)."""
    lon, lat = _transformer(ds).transform(x, y)
    return float(lat), float(lon % 360.0)


def pixel_to_site(ds, row, col, band=1):
    """Latitude, longitude and elevation (m) at the centre of pixel (row, col)."""
    x, y = ds.xy(row, col)                       # pixel centre, projected meters
    lat, lon = xy_to_lonlat(ds, x, y)
    elev = float(ds.read(band)[row, col])
    return {"lat": lat, "lon": lon, "elev_m": elev, "x": x, "y": y,
            "row": row, "col": col}


def highest_point(ds, band=1, margin_px=0):
    z = ds.read(band).astype(float)
    if ds.nodata is not None:
        z = np.where(z == ds.nodata, -np.inf, z)
    z = np.where(np.isfinite(z), z, -np.inf)
    if margin_px > 0:
        z[:margin_px, :] = -np.inf
        z[-margin_px:, :] = -np.inf
        z[:, :margin_px] = -np.inf
        z[:, -margin_px:] = -np.inf
    row, col = np.unravel_index(np.argmax(z), z.shape)
    return pixel_to_site(ds, int(row), int(col), band)


def site_at(path, x=None, y=None, row=None, col=None):
    """Open a tile and return the site at projected (x, y) or at pixel (row, col)."""
    with rasterio.open(path) as ds:
        if x is not None:
            row, col = ds.index(x, y)
        return pixel_to_site(ds, row, col)