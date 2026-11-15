"""
Pick a real site from a PGDA terrain tile and plot its Sun/Earth elevation.
Edit TILE to the real filename of the Site 04 '*_surf.tif'.
"""
import os
import sys

import matplotlib.pyplot as plt
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ephemeris as eph
import sites

print("using:", sites.__file__)

TILE = os.path.join(HERE, ".", "data", "Site04_final_adj_5mpp_surf.tif")   # <- real filename
OBSERVER_M = 2.0                       # observer height above the ground

with rasterio.open(TILE) as ds:
    site = sites.highest_point(ds, margin_px=400)     # highest valid pixel in the tile

print("site: lat %.4f  lon %.4f  ground elevation %.1f m"
      % (site["lat"], site["lon"], site["elev_m"]))
print("tile pixel (row, col):", site["row"], site["col"], " x, y (m):", site["x"], site["y"])

eph.load_kernels()
ets = eph.make_times("2026-11-01T00:00:00", days=365, step_hours=6)
height = site["elev_m"] + OBSERVER_M
res = eph.sun_earth_azel(site["lat"], site["lon"], height, ets)
days = (ets - ets[0]) / 86400.0

for name, (az, el) in res.items():
    print(name, "elevation range (deg):", round(el.min(), 2), round(el.max(), 2))

plt.plot(days, res["sun"][1], label="Sun")
plt.plot(days, res["earth"][1], label="Earth")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("days from start")
plt.ylabel("elevation above a flat horizon (deg)")
plt.legend()
plt.show()