from netCDF4 import Dataset
import numpy as np
from osgeo import osr, gdal
import time as t
import os


def get_pro_(dataset):
    projection_var_name = "goes_imager_projection"

    # Retrieve projection attributes
    if projection_var_name in dataset.variables:
        projection_var = dataset.variables[projection_var_name]
        proj_params = projection_var.__dict__
    else:
        proj_params = dataset.__dict__

    # Example: Constructing a Proj4 string based on typical parameters
    proj4_string = "+proj=geos"
    if "semi_major_axis" in proj_params:
        proj4_string += f" +a={proj_params['semi_major_axis']}"
    if "inverse_flattening" in proj_params:
        inverse_flattening = proj_params["inverse_flattening"]
        f = 1 / inverse_flattening
        proj4_string += f" +f={f}"
    if "longitude_of_projection_origin" in proj_params:
        proj4_string += f" +lon_0={proj_params['longitude_of_projection_origin']}"
    if "latitude_of_projection_origin" in proj_params:
        proj4_string += f" +lat_0={proj_params['latitude_of_projection_origin']}"
    if "perspective_point_height" in proj_params:
        proj4_string += f" +h={proj_params['perspective_point_height']}"
    if "sweep_angle_axis" in proj_params:
        proj4_string += f" +sweep={proj_params['sweep_angle_axis']}"

    # Add any other necessary Proj4 parameters as needed

    print(f"Constructed source Proj4 String: {proj4_string}")

    # Create a spatial reference object and import the constructed Proj4 string
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromProj4(proj4_string)

    dataset.close()
    return spatial_ref


def exportImage(image, path):
    driver = gdal.GetDriverByName("GTiff")
    return driver.CreateCopy(path, image, 0)


def getGeoT(extent, nlines, ncols):
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3], 0, -resy]


def getScaleOffset(nc):
    scale = nc.variables[var_name].getncattr("scale_factor", 1.0)
    offset = nc.variables[var_name].getncattr("add_offset", 0.0)
    return scale, offset


def convert_nc_to_tif(nc_file, output_path, dstSRS):
    var_name = "LST"
    dataset = Dataset(nc_file, mode="r")
    custom_projection = get_pro_(dataset)
    band = gdal.Open(f'NETCDF:"{nc_file}":{var_name}')
    gdal.Translate("./tmp_testing_python.tif", band, outputSRS=custom_projection)
    gdal.Warp(output_path, "./tmp_testing_python.tif", dstSRS=dstSRS)


def process_directory(input_base_path, output_base_path, targetPrj, resolution=2.0):
    for day in range(120, 121):
        day_dir = f"{day:03d}"
        day_input_path = os.path.join(input_base_path, day_dir)
        output_day_path = os.path.join(output_base_path, day_dir)

        if not os.path.exists(day_input_path):
            print(f"Skipping missing day directory: {day_input_path}")
            continue

        os.makedirs(output_day_path, exist_ok=True)

        for hour in range(24):
            hour_dir = f"{hour:02d}"
            hour_input_path = os.path.join(day_input_path, hour_dir)

            if not os.path.exists(hour_input_path):
                print(f"Skipping missing hour directory: {hour_input_path}")
                continue

            nc_files = [f for f in os.listdir(hour_input_path) if f.endswith(".nc")]
            if not nc_files:
                print(f"No .nc files found in {hour_input_path}. Skipping...")
                continue

            nc_file = os.path.join(hour_input_path, nc_files[0])
            output_path = os.path.join(output_day_path, f"{hour_dir}.tif")
            print(f"Processing {nc_file} => {output_path}")
            convert_nc_to_tif(nc_file, output_path, targetPrj)


# def remap(path, resolution, var_name, driver="NETCDF"):
#     nc = Dataset(path, mode="r")
#     scale, offset = getScaleOffset(nc)
#     geo_extent = nc.variables["geospatial_lat_lon_extent"]
#     max_lon = float(geo_extent.geospatial_westbound_longitude)
#     min_lon = float(geo_extent.geospatial_eastbound_longitude)
#     min_lat = float(geo_extent.geospatial_southbound_latitude)
#     max_lat = float(geo_extent.geospatial_northbound_latitude)
#     extent = [min_lon, min_lat, max_lon, max_lat]
#     print("extent:", extent)

#     #     -5434894.885056,  # lower-left x
#     #     -5434894.885056,  # lower-left y
#     #     5434894.885056,  # upper-right x
#     #     5434894.885056,  # upper-right y

#     H = nc.variables["goes_imager_projection"].perspective_point_height
#     print("nc.variables['x_image_bounds'][0]: ", nc.variables["x_image_bounds"][0])
#     x1 = nc.variables["x_image_bounds"][0] * H
#     x2 = nc.variables["x_image_bounds"][1] * H
#     y1 = nc.variables["y_image_bounds"][1] * H
#     y2 = nc.variables["y_image_bounds"][0] * H
#     GOES17_EXTENT = [x1, y1, x2, y2]
#     nc.close()
#     if driver == "NETCDF":
#         connectionInfo = f'NETCDF:"{path}":{var_name}'
#     else:
#         connectionInfo = f'HDF5:"{path}"://{var_name}'

#     raw = gdal.Open(connectionInfo, gdal.GA_ReadOnly)

#     if not raw:
#         raise ValueError(f"Failed to open file {path}")

#     print("Setting source projection and geo-transform")
#     raw.SetProjection(sourcePrj.ExportToWkt())
#     raw.SetGeoTransform(getGeoT(GOES17_EXTENT, raw.RasterYSize, raw.RasterXSize))

#     print("extent[2] - extent[0]:", extent[2] - extent[0])
#     sizex = int(((extent[2] - extent[0]) * KM_PER_DEGREE) / resolution)
#     sizey = int(((extent[3] - extent[1]) * KM_PER_DEGREE) / resolution)

#     print(f"Extent width (degrees): {extent[2] - extent[0]}")
#     print(f"Extent height (degrees): {extent[3] - extent[1]}")
#     print(f"Grid dimensions: sizex={sizex}, sizey={sizey}")

#     if sizex <= 0 or sizey <= 0:
#         raise ValueError(f"Invalid grid dimensions: sizex={sizex}, sizey={sizey}")

#     memDriver = gdal.GetDriverByName("MEM")

#     if not memDriver:
#         raise RuntimeError("Memory driver not available")

#     grid = memDriver.Create("grid", sizex, sizey, 1, gdal.GDT_Float32)

#     if not grid:
#         raise RuntimeError("Failed to create in-memory grid")

#     print("Setting target projection and geo-transform for grid")
#     grid.SetProjection(targetPrj.ExportToWkt())
#     grid.SetGeoTransform(getGeoT(extent, grid.RasterYSize, grid.RasterXSize))

#     print(f"Remapping {path} for variable {var_name}")
#     start = t.time()

#     gdal.ReprojectImage(
#         raw,
#         grid,
#         sourcePrj.ExportToWkt(),
#         targetPrj.ExportToWkt(),
#         gdal.GRA_NearestNeighbour,
#     )

#     raw = None

#     array = grid.ReadAsArray()
#     array = np.ma.masked_where(array == -1, array)
#     array = array * scale + offset

#     grid.GetRasterBand(1).SetNoDataValue(-1)
#     grid.GetRasterBand(1).WriteArray(array)
#     array = np.ma.masked_where(array == -1, array)

#     return grid


if __name__ == "__main__":
    KM_PER_DEGREE = 111.32  # satellite km per degree
    # nc_file = "/s/parsons/b/others/sustain/varsh/Python/GOES/OR_ABI-L2-LSTC-M6_G16_s20220010001173_e20220010003546_c20220010005176.nc"

    input_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/data/noaa-goes16/ABI-L2-LSTC/2022/"
    )
    output_base_path = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/TifFolder/"
    )

    # targetPrj = osr.SpatialReference()
    # targetPrj.ImportFromEPSG(4326)
    targetPrj = osr.SpatialReference()
    targetPrj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    print(f"Target Projection: {targetPrj.ExportToProj4()}")

    var_name = "LST"

    resolution = 2.0

    # output_path = "/s/parsons/b/others/sustain/varsh/Python/GOES/LSTC_output.tif"
    # convert_nc_to_tif(nc_file, output_path, targetPrj)

    process_directory(input_base_path, output_base_path, targetPrj)

    # exportImage(grid, output_path)
