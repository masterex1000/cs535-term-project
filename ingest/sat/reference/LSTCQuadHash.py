import socket
import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile, fiona
import geopy.distance
import os
from osgeo import osr
import json
from pyquadkey2.quadkey import QuadKey
import pyquadkey2
import numpy as np
import rasterio
from shapely.geometry import box
from fiona.crs import from_string


# Get quadhash tile of a given coordinate
def get_quad_tile(lat, lon, precision):
    ret = mercantile.tile(lon, lat, precision)
    return ret


def get_quad_key_from_tile(x, y, zoom):
    return mercantile.quadkey(x, y, zoom)


# Given a quad_key, get the corresponding quadtile
def get_tile_from_key(key):
    return mercantile.quadkey_to_tile(key)


# Get the quadhash string from lat, long and quadhash length
def get_quad_key(lat, lon, zoom):
    tile = get_quad_tile(lat, lon, precision=zoom)
    return get_quad_key_from_tile(tile.x, tile.y, tile.z)


# Given a box, find all tiles that lie inside that coordinate box
def find_all_inside_box(lat1, lat2, lon1, lon2, zoom):
    all_tiles, all_tiles_quadhash, bounding_boxes = [], [], []
    top_left_quad_tile = get_quad_tile(lat1, lon1, zoom)
    bottom_right_quad_tile = get_quad_tile(lat2, lon2, zoom)

    x1 = top_left_quad_tile.x
    x2 = bottom_right_quad_tile.x
    y1 = top_left_quad_tile.y
    y2 = bottom_right_quad_tile.y

    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            all_tiles.append(mercantile.Tile(x=i, y=j, z=zoom))
            info = mercantile.Tile(x=i, y=j, z=zoom)
            qk = get_quad_key_from_tile(info.x, info.y, info.z)
            all_tiles_quadhash.append(qk)
            bounding_boxes.append(get_bounding_lng_lat(qk))
    return all_tiles_quadhash, bounding_boxes


# Given a quadtile, get its lat/long bounds
def get_bounding_lng_lat(tile_key):
    tile = get_tile_from_key(tile_key)
    bounds = mercantile.bounds(tile)
    return [bounds.west, bounds.east, bounds.north, bounds.south]


def create_shp_file_shri(zoom=None):
    states = gpd.read_file(
        "/s/parsons/b/others/sustain/diurnalModel/data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
    )
    states = states.to_crs("EPSG:4326")
    states["NAME"] = states["NAME"].str.replace(" ", "_")

    schema = {"geometry": "Polygon", "properties": {"Quadkey": "str"}}
    if zoom is None:
        zoom = [12, 14]

    for ind, row in states.iterrows():
        state_name = row["NAME"]
        state_info = states[states["NAME"] == state_name]

        # Extract bounding box coordinates for the state
        area_of_interest_lat1, area_of_interest_lat2 = (
            state_info["geometry"].bounds["maxy"].iloc[0],
            state_info["geometry"].bounds["miny"].iloc[0],
        )
        area_of_interest_lon1, area_of_interest_lon2 = (
            state_info["geometry"].bounds["minx"].iloc[0],
            state_info["geometry"].bounds["maxx"].iloc[0],
        )

        for z in zoom:
            directory = os.path.join("./states_quads(9)", f"quadshape_{z}_{state_name}")
            os.makedirs(directory, exist_ok=True)
            shapefile_path = os.path.join(directory, "quadhash.shp")

            with fiona.open(
                shapefile_path,
                mode="w",
                driver="ESRI Shapefile",
                schema=schema,
                crs=from_string("+datum=WGS84 +ellps=WGS84 +no_defs +proj=longlat"),
            ) as polyShp:
                quads, bounds = find_all_inside_box(
                    area_of_interest_lat1,
                    area_of_interest_lat2,
                    area_of_interest_lon1,
                    area_of_interest_lon2,
                    zoom=z,
                )
                count = 0
                for i in range(len(quads)):
                    smaller_region_box = box(
                        bounds[i][0], bounds[i][3], bounds[i][1], bounds[i][2]
                    )

                    if state_info["geometry"].iloc[0].intersects(
                        smaller_region_box
                    ) or state_info["geometry"].iloc[0].contains(smaller_region_box):
                        count += 1
                        xyList = [
                            (bounds[i][0], bounds[i][2]),
                            (bounds[i][1], bounds[i][2]),
                            (bounds[i][1], bounds[i][3]),
                            (bounds[i][0], bounds[i][3]),
                            (bounds[i][0], bounds[i][2]),
                        ]
                        rowDict = {
                            "geometry": {"type": "Polygon", "coordinates": [xyList]},
                            "properties": {"Quadkey": quads[i]},
                        }
                        polyShp.write(rowDict)
                        print(f"running for {state_name},{i}, {len(quads)}")
                        # print(rowDict)
                        # print(f", hahaha, {i}")
                print(
                    f"Shape file generated successfully with no. of rows: {count} at zoom level: {z} for state: {state_name}"
                )


if __name__ == "__main__":
    create_shp_file_shri(zoom=[9])
    # nm = gpd.read_file('/s/parsons/b/others/sustain/diurnalModel/data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    # print(nm.head(61))
