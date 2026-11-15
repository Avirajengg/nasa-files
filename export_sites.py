"""
Batch export for the web app. For every *_surf.tif tile under data/ this picks
a site (highest interior pixel), computes the terrain horizon at several
observer heights, computes Sun/Earth azimuth and elevation for ~2 years, and
writes everything to data/sites.json.

Run:  python export_sites.py            (finds data/ next to this script or one level up)
      python export_sites.py C:/path/to/data
"""
import glob
import json
import os
import re
import sys

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "src")):
    sys.path.insert(0, p)

import ephemeris as eph
import horizon as hz
import metrics as mt
import sites

START, DAYS, STEP_H = "2026-11-15T00:00:00", 730, 2.0
HEIGHTS = [2,5, 10, 20]            # observer heights above ground (m)
# Labels from the PGDA README; check them before you publish.
NAMES = {"01": "Connecting ridge", "04": "Shackleton rim",
         "11": "de Gerlache rim", "23": "Malapert massif"}


def q(a, scale=100):
    """Round to integers (value * scale) so the JSON stays small."""
    return np.round(np.asarray(a) * scale).astype(int).tolist()


def summarize(res, r_sun, r_earth, hor):
    saz, sel = res["sun"]
    eaz, eel = res["earth"]
    c = hz.visible(saz, sel, hor, r_sun, "center")
    a = hz.visible(saz, sel, hor, r_sun, "any")
    f = hz.visible(saz, sel, hor, r_sun, "full")
    ec = hz.visible(eaz, eel, hor, r_earth, "center")
    ea = hz.visible(eaz, eel, hor, r_earth, "any")
    return {
        "sun_center_pct": round(100 * mt.fraction_true(c), 2),
        "sun_any_pct": round(100 * mt.fraction_true(a), 2),
        "sun_full_pct": round(100 * mt.fraction_true(f), 2),
        "sun_longest_dark_h_center": mt.longest_run_hours(c, STEP_H, value=False),
        "sun_longest_dark_h_any": mt.longest_run_hours(a, STEP_H, value=False),
        "earth_center_pct": round(100 * mt.fraction_true(ec), 2),
        "earth_any_pct": round(100 * mt.fraction_true(ea), 2),
    }


def main(data_dir=None):
    if data_dir is None:
        data_dir = next(d for d in (os.path.join(HERE, "data"),
                                    os.path.join(HERE, "..", "data"))
                        if os.path.isdir(d))
    tiles = sorted(glob.glob(os.path.join(data_dir, "**", "*_surf.tif"), recursive=True))
    if not tiles:
        raise SystemExit(f"No *_surf.tif tiles found under {data_dir}")
    print(f"{len(tiles)} tiles found")

    eph.load_kernels()
    ets = eph.make_times(START, DAYS, STEP_H)
    r_sun = eph.body_angular_radius("SUN", ets)
    r_earth = eph.body_angular_radius("EARTH", ets)

    out = {"meta": {"start_utc": START, "step_hours": STEP_H, "n": int(len(ets)),
                    "heights_m": HEIGHTS, "az_step_deg": 1,
                    "scale": 100, "radius_scale": 1000,
                    "sun_radius": q(r_sun, 1000), "earth_radius": q(r_earth, 1000)},
           "sites": []}

    for path in tiles:
        m = re.search(r"[Ss]ite(\d+)", os.path.basename(path))
        sid = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
        with rasterio.open(path) as ds:
            z, T, nodata = ds.read(1), ds.transform, ds.nodata
            margin = min(400, min(z.shape) // 4)
            site = sites.highest_point(ds, margin_px=margin)
        res = eph.sun_earth_azel(site["lat"], site["lon"], site["elev_m"] + 2.0, ets)

        horizons, summaries = {}, {}
        for h in HEIGHTS:
            hor = hz.compute_horizon(z, T, site["x"], site["y"], site["lon"],
                                     site["elev_m"] + h, nodata=nodata)
            hor["horizon_deg"] = np.nan_to_num(hor["horizon_deg"], nan=-90.0)
            horizons[str(h)] = q(hor["horizon_deg"])
            summaries[str(h)] = summarize(res, r_sun, r_earth, hor)
            rng = float(np.median(hor["range_m"]))
        entry = {
            "id": sid, "name": NAMES.get(sid, f"Site {sid}"),
            "lat": round(site["lat"], 5), "lon": round(site["lon"], 5),
            "elev_m": round(site["elev_m"], 1), "tile_range_median_m": round(rng),
            "horizon": horizons, "summary": summaries,
            "sun_az": q(res["sun"][0]), "sun_el": q(res["sun"][1]),
            "earth_az": q(res["earth"][0]), "earth_el": q(res["earth"][1]),
        }
        out["sites"].append(entry)
        s2 = summaries["2"]
        print(f"Site {sid}: lat {entry['lat']} lon {entry['lon']} elev {entry['elev_m']} m | "
              f"Sun {s2['sun_center_pct']}% Earth {s2['earth_center_pct']}%")

    path = os.path.join(data_dir, "sites.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)