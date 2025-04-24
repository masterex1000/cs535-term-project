from netCDF4 import Dataset
import numpy as np
from osgeo import osr, gdal
import time as t

# Define constants
KM_PER_DEGREE = 111.32
GOES16_EXTENT = [-5434894.885056, -5434894.885056, 5434894.885056, 5434894.885056]

# Define spatial reference systems
sourcePrj = osr.SpatialReference()
sourcePrj.ImportFromProj4(
    "+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.00335281068119356027 +lat_0=0.0 +lon_0=-89.5 +sweep=x +no_defs"
)

targetPrj = osr.SpatialReference()
targetPrj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


# Define functions
def getGeoT(extent, nlines, ncols):
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3], 0, -resy]


def getScaleOffset(path):
    nc = Dataset(path, mode="r")
    scale = nc.variables["CMI_C14"].scale_factor
    offset = nc.variables["CMI_C14"].add_offset
    nc.close()
    return scale, offset


def convertLST(T14, T15, theta=0.0):
    C = 1.0
    A1 = 0.5
    A2 = 0.1
    A3 = 0.2
    D = 0.5

    Ts = (
        T14
        + C * (T14 - T15)
        + A1 * (T14 - T15) * np.exp(-A2 / T14) * (1 - np.exp(-A3 / T15))
        + D * (1 / np.cos(np.deg2rad(theta)) - 1)
    )
    return Ts


def remap(path, extent, resolution, driver):
    scale, offset = getScaleOffset(path)

    if driver == "NETCDF":
        cmi_c14_info = 'NETCDF:"' + path + '":CMI_C14'
        cmi_c15_info = 'NETCDF:"' + path + '":CMI_C15'
    else:  # HDF5
        cmi_c14_info = 'HDF5:"' + path + '"://CMI_C14'
        cmi_c15_info = 'HDF5:"' + path + '"://CMI_C15'

    raw_c14 = gdal.Open(cmi_c14_info, gdal.GA_ReadOnly)
    raw_c15 = gdal.Open(cmi_c15_info, gdal.GA_ReadOnly)

    raw_c14.SetProjection(sourcePrj.ExportToWkt())
    raw_c14.SetGeoTransform(
        getGeoT(GOES16_EXTENT, raw_c14.RasterYSize, raw_c14.RasterXSize)
    )
    raw_c15.SetProjection(sourcePrj.ExportToWkt())
    raw_c15.SetGeoTransform(
        getGeoT(GOES16_EXTENT, raw_c15.RasterYSize, raw_c15.RasterXSize)
    )

    sizex = int(((extent[2] - extent[0]) * KM_PER_DEGREE) / resolution)
    sizey = int(((extent[3] - extent[1]) * KM_PER_DEGREE) / resolution)

    memDriver = gdal.GetDriverByName("MEM")
    grid_c14 = memDriver.Create("grid_c14", sizex, sizey, 1, gdal.GDT_Float32)
    grid_c15 = memDriver.Create("grid_c15", sizex, sizey, 1, gdal.GDT_Float32)
    grid_c14.SetProjection(targetPrj.ExportToWkt())
    grid_c14.SetGeoTransform(
        getGeoT(extent, grid_c14.RasterYSize, grid_c14.RasterXSize)
    )
    grid_c15.SetProjection(targetPrj.ExportToWkt())
    grid_c15.SetGeoTransform(
        getGeoT(extent, grid_c15.RasterYSize, grid_c15.RasterXSize)
    )

    print("Remapping", path)
    start = t.time()
    gdal.ReprojectImage(
        raw_c14,
        grid_c14,
        sourcePrj.ExportToWkt(),
        targetPrj.ExportToWkt(),
        gdal.GRA_NearestNeighbour,
        options=["NUM_THREADS=ALL_CPUS"],
    )
    gdal.ReprojectImage(
        raw_c15,
        grid_c15,
        sourcePrj.ExportToWkt(),
        targetPrj.ExportToWkt(),
        gdal.GRA_NearestNeighbour,
        options=["NUM_THREADS=ALL_CPUS"],
    )
    print("- finished! Time:", t.time() - start, "seconds")

    raw_c14 = None
    raw_c15 = None

    array_c14 = grid_c14.ReadAsArray()
    array_c15 = grid_c15.ReadAsArray()

    np.ma.masked_where(array_c14, array_c14 == -1, False)
    np.ma.masked_where(array_c15, array_c15 == -1, False)

    lst_band = convertLST(array_c14, array_c15)
    lst_band[lst_band > 500] = 65535.0

    grid_c14.GetRasterBand(1).SetNoDataValue(-1)
    grid_c14.GetRasterBand(1).WriteArray(array_c14)

    grid_c15.GetRasterBand(1).SetNoDataValue(-1)
    grid_c15.GetRasterBand(1).WriteArray(array_c15)

    return lst_band, grid_c14


def save_as_tiff(array, grid, output_path):
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, grid.RasterXSize, grid.RasterYSize, 1, gdal.GDT_Float32
    )
    out_ds.SetProjection(grid.GetProjection())
    out_ds.SetGeoTransform(grid.GetGeoTransform())
    out_ds.GetRasterBand(1).WriteArray(array)
    out_ds.GetRasterBand(1).SetNoDataValue(-1)
    out_ds.FlushCache()
    out_ds = None


# Path to the NetCDF file
file_path = "/s/lattice-151/a/all/all/all/sustain/data/noaa-goes16/ABI-L2-MCMIPC/2022/001/00/OR_ABI-L2-MCMIPC-M6_G16_s20220010001173_e20220010003557_c20220010004046.nc"

# Define the extent and resolution for the remapping
extent = [-130.0, 20.0, -60.0, 55.0]  # example extent in degrees
resolution = 0.018  # example resolution in degrees

# Perform the remapping
lst_band, grid_c14 = remap(file_path, extent, resolution, "NETCDF")

# Save the remapped LST band as a TIFF file
tiff_output_path = "/s/parsons/b/others/sustain/varsh/Python/GOES/tmp_LST.tif"
save_as_tiff(lst_band, grid_c14, tiff_output_path)

print(f"TIFF file saved at: {tiff_output_path}")
