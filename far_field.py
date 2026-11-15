"""
Far-field check: does terrain OUTSIDE a site's 16 km tile change the result?

Reads the PGDA 87-90S mosaic (ldem_87s_5mpp.tif, ~3.3 GB, same polar stereographic
map as the site tiles) with windowed, downsampled reads, computes a long-range
horizon (default 40 km, 20 m cells), merges it with the 5 m tile horizon already in
data/sites.json (taking the higher of the two in every direction) and prints
before/after visibility. Only sites poleward of 87 S are covered by the mosaic
(Sites 01, 04, 07, 11); the others are skipped.

Run:  python far_field.py [path/to/ldem_87s_5mpp.tif] [site ids ...]
Output: data/sites_farfield.json  (same format as sites.json, merged horizons)
"""
import json
import math
import os
import sys

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "src")):
    sys.path.insert(0, p)

import horizon as hz
import metrics as mt

RADIUS_M, COARSE_M = 40_000.0, 20.0


def far_horizon(ds, x0, y0, lon_deg, site_height_m, radius_m=RADIUS_M, coarse_m=COARSE_M):
    """Horizon out to radius_m from a big raster, read as a downsampled window."""
    res = ds.res[0]
    win = from_bounds(x0 - radius_m, y0 - radius_m, x0 + radius_m, y0 + radius_m, ds.transform)
    c0, r0 = max(0, math.floor(win.col_off)), max(0, math.floor(win.row_off))
    c1 = min(ds.width, math.ceil(win.col_off + win.width))
    r1 = min(ds.height, math.ceil(win.row_off + win.height))
    win = Window(c0, r0, c1 - c0, r1 - r0)
    f = max(1, int(round(coarse_m / res)))
    out_h, out_w = max(1, int(win.height // f)), max(1, int(win.width // f))
    z = ds.read(1, window=win, out_shape=(out_h, out_w),
                resampling=Resampling.average).astype("float32")
    t = ds.window_transform(win) * Affine.scale(win.width / out_w, win.height / out_h)
    return hz.compute_horizon(z, t, x0, y0, lon_deg, site_height_m,
                              step_m=abs(t.a), max_range_m=radius_m, nodata=ds.nodata)


def summarize(sa, se, ea, ee, rs, re, horizon, step_h):
    c = hz.visible(sa, se, horizon, rs, "center")
    a = hz.visible(sa, se, horizon, rs, "any")
    f = hz.visible(sa, se, horizon, rs, "full")
    ec = hz.visible(ea, ee, horizon, re, "center")
    ea_ = hz.visible(ea, ee, horizon, re, "any")
    return {"sun_center_pct": round(100 * mt.fraction_true(c), 2),
            "sun_any_pct": round(100 * mt.fraction_true(a), 2),
            "sun_full_pct": round(100 * mt.fraction_true(f), 2),
            "sun_longest_dark_h_center": mt.longest_run_hours(c, step_h, value=False),
            "sun_longest_dark_h_any": mt.longest_run_hours(a, step_h, value=False),
            "sun_longest_lit_h_center": mt.longest_run_hours(c, step_h, value=True),
            "sun_longest_lit_h_any": mt.longest_run_hours(a, step_h, value=True),
            "earth_center_pct": round(100 * mt.fraction_true(ec), 2),
            "earth_any_pct": round(100 * mt.fraction_true(ea_), 2)}


def main(mosaic, only=None, data_dir=None):
    data_dir = data_dir or next(d for d in (os.path.join(HERE, "data"),
                                            os.path.join(HERE, "..", "data")) if os.path.isdir(d))
    j = json.load(open(os.path.join(data_dir, "sites.json")))
    m = j["meta"]; step = m["step_hours"]
    rs = np.array(m["sun_radius"]) / m["radius_scale"]
    re = np.array(m["earth_radius"]) / m["radius_scale"]
    az = np.arange(360.0)
    kept = []
    with rasterio.open(mosaic) as ds:
        crs = CRS.from_wkt(ds.crs.to_wkt())
        fwd = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        for s in j["sites"]:
            if only and s["id"] not in only:
                continue
            x0, y0 = fwd.transform(s["lon"], s["lat"])
            if not (ds.bounds.left < x0 < ds.bounds.right and ds.bounds.bottom < y0 < ds.bounds.top):
                print(f"Site {s['id']}: outside the mosaic (needs 87-90 S), skipped")
                continue
            sa, se = np.array(s["sun_az"]) / 100, np.array(s["sun_el"]) / 100
            ea, ee = np.array(s["earth_az"]) / 100, np.array(s["earth_el"]) / 100
            for h in m["heights_m"]:
                near = np.array(s["horizon"][str(h)]) / 100
                far = far_horizon(ds, x0, y0, s["lon"], s["elev_m"] + h)
                merged = np.maximum(near, np.nan_to_num(far["horizon_deg"], nan=-90.0))
                before = summarize(sa, se, ea, ee, rs, re, {"az_deg": az, "horizon_deg": near}, step)
                after = summarize(sa, se, ea, ee, rs, re, {"az_deg": az, "horizon_deg": merged}, step)
                if h in (2, 5):
                    print(f"Site {s['id']} {s['name']:<18} h={h:>2} m | max horizon change "
                          f"{(merged - near).max():5.2f} deg | Sun lit {before['sun_center_pct']:5.1f}% -> "
                          f"{after['sun_center_pct']:5.1f}% | longest lit d {before['sun_longest_lit_h_center']/24:5.0f} -> "
                          f"{after['sun_longest_lit_h_center']/24:5.0f} | longest dark h {before['sun_longest_dark_h_center']:5.0f} -> "
                          f"{after['sun_longest_dark_h_center']:5.0f} | Earth {before['earth_center_pct']:5.1f}% -> "
                          f"{after['earth_center_pct']:5.1f}%")
                s["horizon"][str(h)] = np.round(merged * 100).astype(int).tolist()
                s["summary"][str(h)] = after
            s["tile_range_median_m"] = int(RADIUS_M)
            s["horizon_source"] = "5 m tile + mosaic far field (%d km, %d m cells)" % (RADIUS_M / 1000, COARSE_M)
            kept.append(s)
    j["sites"] = kept
    out = os.path.join(data_dir, "sites_farfield.json")
    json.dump(j, open(out, "w"), separators=(",", ":"))
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    mosaic = args[0] if args else os.path.join(HERE, "data", "ldem_87s_5mpp.tif")
    main(mosaic, set(args[1:]) or None)