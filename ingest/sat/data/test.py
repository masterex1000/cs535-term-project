from osgeo import gdal

# Open the source dataset
ds = gdal.Open("00.tif")

# Bounding box for Florida in WGS84 (lon/lat)
# Slightly extended for safety
window = [-88.0, 31.5, -79.5, 24.0]

# Translate (crop)
cropped_ds = gdal.Translate(
    "florida.tif",
    ds,
    projWin=window,
    format="GTiff"
)

# Close datasets
cropped_ds = None
ds = None
