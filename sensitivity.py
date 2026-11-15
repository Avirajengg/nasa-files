"""
Sun-disk and observer-height sensitivity for one site on a PGDA tile.

For each observer height above the ground it reports how much of the year the
Sun and Earth are visible, with the Sun/Earth treated as a point (centre above
the horizon), as a disk with any part visible, and as a disk fully visible.
Edit TILE_NAME if your file differs.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "src")):
    sys.path.insert(0, p)

import ephemeris as eph
import horizon as hz
import metrics as mt
import sites

TILE_NAME = "Site04_final_adj_5mpp_surf.tif"
HEIGHTS_M = [0.0, 2.0, 10.0, 20.0]          # observer height above the ground
START, DAYS, STEP_H = "2026-11-01T00:00:00", 365, 1.0

tile = next(p for p in (os.path.join(HERE, "data", TILE_NAME),
                        os.path.join(HERE, "..", "data", TILE_NAME))
            if os.path.exists(p))
with rasterio.open(tile) as ds:
    z = ds.read(1)
    transform, nodata = ds.transform, ds.nodata
    site = sites.highest_point(ds, margin_px=400)
print("site: lat %.4f  lon %.4f  ground %.1f m" % (site["lat"], site["lon"], site["elev_m"]))

eph.load_kernels()
ets = eph.make_times(START, DAYS, STEP_H)
# Sun/Earth directions barely change over a 20 m height difference, so compute once.
res = eph.sun_earth_azel(site["lat"], site["lon"], site["elev_m"] + 2.0, ets)
r_sun = eph.body_angular_radius("SUN", ets)
r_earth = eph.body_angular_radius("EARTH", ets)
print("Sun radius %.3f-%.3f deg, Earth radius %.3f-%.3f deg"
      % (r_sun.min(), r_sun.max(), r_earth.min(), r_earth.max()))

head = ("height", "Sun pt%", "Sun any%", "Sun full%", "dark h (pt)", "dark h (any)",
        "Earth pt%", "Earth any%")
print("\n" + "".join(f"{h:>13}" for h in head))
rows = []
for h in HEIGHTS_M:
    hor = hz.compute_horizon(z, transform, site["x"], site["y"], site["lon"],
                             site["elev_m"] + h, nodata=nodata)
    saz, sel = res["sun"]
    eaz, eel = res["earth"]
    sun_pt = hz.visible(saz, sel, hor, r_sun, "center")
    sun_any = hz.visible(saz, sel, hor, r_sun, "any")
    sun_full = hz.visible(saz, sel, hor, r_sun, "full")
    ea_pt = hz.visible(eaz, eel, hor, r_earth, "center")
    ea_any = hz.visible(eaz, eel, hor, r_earth, "any")
    row = (h, 100 * mt.fraction_true(sun_pt), 100 * mt.fraction_true(sun_any),
           100 * mt.fraction_true(sun_full),
           mt.longest_run_hours(sun_pt, STEP_H, value=False),
           mt.longest_run_hours(sun_any, STEP_H, value=False),
           100 * mt.fraction_true(ea_pt), 100 * mt.fraction_true(ea_any))
    rows.append(row)
    print("".join(f"{v:>13.1f}" for v in row))

rows = np.array(rows)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rows[:, 0], rows[:, 1], "o-", label="Sun (centre)")
ax.plot(rows[:, 0], rows[:, 2], "o-", label="Sun (any part)")
ax.plot(rows[:, 0], rows[:, 3], "o-", label="Sun (fully)")
ax.set_xlabel("observer height above ground (m)")
ax.set_ylabel("% of year lit")
ax.legend()
plt.tight_layout(); plt.show()