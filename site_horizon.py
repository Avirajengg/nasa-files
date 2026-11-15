"""
Terrain-aware Sun and Earth visibility for a site picked from a PGDA tile.
Prints percent lit, longest darkness and Earth-visible fraction, flat horizon
versus real terrain, and draws a sky plot. Edit TILE_NAME if yours differs.
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
OBSERVER_M = 2.0
START, DAYS, STEP_H = "2026-11-01T00:00:00", 365, 1.0

tile = next(p for p in (os.path.join(HERE, "data", TILE_NAME),
                        os.path.join(HERE, "..", "data", TILE_NAME))
            if os.path.exists(p))
with rasterio.open(tile) as ds:
    site = sites.highest_point(ds, margin_px=400)
    hor = hz.horizon_for_site(ds, site, observer_m=OBSERVER_M)

print("site: lat %.4f  lon %.4f  ground %.1f m" % (site["lat"], site["lon"], site["elev_m"]))
print("horizon: max %.2f deg, mean %.2f deg" % (np.nanmax(hor["horizon_deg"]), np.nanmean(hor["horizon_deg"])))
print("terrain seen: shortest ray %.0f m, median %.0f m (tile limit)"
      % (hor["range_m"].min(), np.median(hor["range_m"])))

eph.load_kernels()
ets = eph.make_times(START, DAYS, STEP_H)
res = eph.sun_earth_azel(site["lat"], site["lon"], site["elev_m"] + OBSERVER_M, ets)

print("\n                      flat horizon   with terrain")
for name in ("sun", "earth"):
    az, el = res[name]
    flat = el > 0
    terr = hz.visible(az, el, hor)
    print("%-5s  visible %%      %6.1f         %6.1f"
          % (name, 100 * mt.fraction_true(flat), 100 * mt.fraction_true(terr)))
    if name == "sun":
        print("sun   longest dark h %6.1f         %6.1f"
              % (mt.longest_run_hours(flat, STEP_H, value=False),
                 mt.longest_run_hours(terr, STEP_H, value=False)))
        sun_vis = terr

# sky plot: horizon profile with Sun and Earth tracks over the year
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3, 1])
ax1.fill_between(hor["az_deg"], hor["horizon_deg"], -10, color="0.7", label="terrain")
ax1.plot(res["sun"][0], res["sun"][1], ".", ms=1, color="tab:orange", label="Sun")
ax1.plot(res["earth"][0], res["earth"][1], ".", ms=1, color="tab:blue", label="Earth")
ax1.set_xlim(0, 360); ax1.set_ylim(-8, 10)
ax1.set_xlabel("azimuth (deg from north)"); ax1.set_ylabel("elevation (deg)")
ax1.legend(markerscale=8, loc="upper right")
days = (ets - ets[0]) / 86400.0
ax2.fill_between(days, 0, sun_vis.astype(int), step="mid", color="tab:orange")
ax2.set_xlabel("days from start"); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["dark", "lit"])
plt.tight_layout(); plt.show()