import os
import geopandas as gpd
import mercantile
import fiona
from shapely.geometry import box
from fiona.crs import CRS
import sys

def get_quad_tile(lat, lon, zoom):
    return mercantile.tile(lon, lat, zoom)

def get_quad_key_from_tile(x, y, zoom):
    return mercantile.quadkey(x, y, zoom)

def get_tile_from_key(key):
    return mercantile.quadkey_to_tile(key)

def get_quad_key(lat, lon, zoom):
    tile = get_quad_tile(lat, lon, zoom)
    return get_quad_key_from_tile(tile.x, tile.y, tile.z)

def find_all_inside_box(lat1, lat2, lon1, lon2, zoom):
    all_tiles_quadhash, bounding_boxes = [], []
    top_left = get_quad_tile(lat1, lon1, zoom)
    bottom_right = get_quad_tile(lat2, lon2, zoom)
    
    for x in range(top_left.x, bottom_right.x + 1):
        for y in range(top_left.y, bottom_right.y + 1):
            quadkey = get_quad_key_from_tile(x, y, zoom)
            bounds = mercantile.bounds(x, y, zoom)
            all_tiles_quadhash.append(quadkey)
            bounding_boxes.append([bounds.west, bounds.east, bounds.north, bounds.south])
    
    return all_tiles_quadhash, bounding_boxes

def create_shp_file(zoom_levels=[9]):
    states = gpd.read_file("/s/parsons/b/others/sustain/varsh/Python/GOES/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
    states = states.to_crs("EPSG:4326")
    states["NAME"] = states["NAME"].str.replace(" ", "_")
    
    schema = {"geometry": "Polygon", "properties": {"Quadkey": "str"}}
    
    for _, row in states.iterrows():
        state_name = row["NAME"]
        state_geom = row["geometry"]
        
        bounds = state_geom.bounds
        lat1, lat2 = bounds[3], bounds[1]
        lon1, lon2 = bounds[0], bounds[2]
        
        for zoom in zoom_levels:
            directory = f"./states_quads/quadshape_{zoom}_{state_name}"
            os.makedirs(directory, exist_ok=True)
            shapefile_path = os.path.join(directory, "quadhash.shp")
            
            try:
                with fiona.open(
                    shapefile_path, mode="w", driver="ESRI Shapefile", schema=schema, 
                    crs=CRS.from_string("+proj=longlat +datum=WGS84 +no_defs")
                ) as shp_file:
                    
                    quads, bounds_list = find_all_inside_box(lat1, lat2, lon1, lon2, zoom)
                    count = 0
                    
                    for quadkey, bounds in zip(quads, bounds_list):
                        tile_box = box(bounds[0], bounds[3], bounds[1], bounds[2])
                        
                        if state_geom.intersects(tile_box) or state_geom.contains(tile_box):
                            shp_file.write({
                                "geometry": {"type": "Polygon", "coordinates": [[
                                    (bounds[0], bounds[2]), (bounds[1], bounds[2]), 
                                    (bounds[1], bounds[3]), (bounds[0], bounds[3]), 
                                    (bounds[0], bounds[2])
                                ]]},
                                "properties": {"Quadkey": quadkey}
                            })
                            count += 1
                    
                    print(f"{state_name} - Zoom {zoom}: {count} tiles written")
            except Exception as e:
                print(f"Error processing {state_name} at Zoom {zoom}: {e}")
                sys.exit(1)  # Exit with error code if a failure occurs
    
    print("Processing complete. Exiting script.")
    sys.exit(0)  

if __name__ == "__main__":
    create_shp_file([9])

