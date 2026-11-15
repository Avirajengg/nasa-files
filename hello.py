import rasterio
with rasterio.open("data/Site04_final_adj_5mpp_surf.tif") as ds:    # real filename
    print(ds.width, ds.height, ds.res)
    print(ds.crs)
    print(ds.bounds)
    mid = ds.read(1)[ds.height // 2, ds.width // 2]
    print("center elevation (m):", mid)