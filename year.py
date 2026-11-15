"""Reproduces the one-year Sun/Earth elevation plot for an example site."""
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))          # so 'data/' is found

import ephemeris as eph

LAT, LON, HEIGHT_M = -89.9, 129.0, 0.0     # example site near the south pole

eph.load_kernels("data")
ets = eph.make_times("2026-11-01T00:00:00", days=365, step_hours=6)
res = eph.sun_earth_azel(LAT, LON, HEIGHT_M, ets)
days = (ets - ets[0]) / 86400.0

for name, (az, el) in res.items():
    print(name, "elevation range (deg):", round(el.min(), 2), round(el.max(), 2))

plt.plot(days, res["sun"][1], label="Sun")
plt.plot(days, res["earth"][1], label="Earth")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("days from start")
plt.ylabel("elevation (deg)")
plt.legend()
plt.show()