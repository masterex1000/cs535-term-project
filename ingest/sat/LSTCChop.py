import os
import socket
import geopandas as gpd
import numpy as np
from osgeo import gdal
from datetime import datetime 
import sqlite3
import argparse

future_weather_query ='''
    SELECT
        site_id,
        (timestamp / 3600) * 3600 AS hour_start,
        AVG(ght) AS avg_ght
    FROM samples
    WHERE timestamp >= ?
        AND timestamp < ? + 6 * 3600
        AND site_id=?
    GROUP BY site_id, hour_start
    ORDER BY site_id, hour_start;
'''

def get_site_current_weather(cursor: sqlite3.Cursor, site_id: str, timestamp):
    current_weather_query = '''
        SELECT temp, wind_speed, wind_direction, ght, humidity FROM samples 
        WHERE timestamp <= ? AND timestamp >= (? - 900) AND site_id = ?
        ORDER BY timestamp DESC LIMIT 1;
    '''

    result = cursor.execute(current_weather_query, (timestamp, timestamp, site_id)).fetchone()

    if not result:
        return None
    
    # return dict(result)
    keys = ['temp', 'wind_speed', 'wind_direction', 'ght', 'humidity']
    return dict(zip(keys, result))

def get_site_future_output(cursor: sqlite3.Cursor, site_id: str, timestamp):
    future_weather_query ='''
        SELECT
            site_id,
            (timestamp / 3600) * 3600 AS hour_start,
            AVG(ght) AS avg_ght
        FROM samples
        WHERE timestamp >= ?
            AND timestamp < ? + 6 * 3600
            AND site_id=?
        GROUP BY site_id, hour_start
        ORDER BY site_id, hour_start;
    '''

    result = cursor.execute(future_weather_query, (timestamp, timestamp, site_id)).fetchall()


    ght_values = tuple(row[2] for row in result)

    return ght_values

def get_timestamp(year, day_of_year, hour):
    dt = datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1, hours=hour)

    return dt.timestamp()


def chop_to_locations(database_path: str, input_base_path, output_base_path):

    sites = {
        "BMS": {"lat": 39.7423, "lon": -105.1785, "elevation": 1828.8},
        "IRRSP": {"lat": 39.7423, "lon": -105.1785, "elevation": 1828.8},
        # "NELHA": {"lat": 19.728144, "lon": -156.058936, "elevation": 4}, # THis is in hawaii. Idk but its all white 
        "NWTC": {"lat": 39.9106, "lon": -105.2347, "elevation": 1855},
        "STAC": {"lat": 39.75685, "lon": -104.62025, "elevation": 1674},
        "UAT": {"lat": 32.22969, "lon": -110.95534, "elevation": 786},
        "ULL": {"lat": 30.20506, "lon": -92.39793, "elevation": 5},
    }

    db = sqlite3.connect(database_path)
    cursor = db.cursor()

    # for day in range(365, 366):
    # for day in range(100, 101):
    # for day in range(366):
    # for day in range(335, 366):
    for day in range(357, 366):
        print(f"Processing day: {day}")
        day_dir = f"{day:03d}"
        day_input_path = os.path.join(input_base_path, day_dir)
        output_day_path = os.path.join(output_base_path, day_dir)

        if not os.path.exists(day_input_path):
            print(f"Skipping missing day directory: {day_input_path}")
            continue

        os.makedirs(output_day_path, exist_ok=True)

        for hour in range(24):  # 00 to 23 hours
        # for hour in range(10,11):  # 00 to 23 hours
            hour_dir = f"{hour:02d}"
            hour_input_path = os.path.join(day_input_path, f"{hour_dir}.tif")
            output_hour_path = os.path.join(output_day_path, hour_dir)

            if not os.path.exists(hour_input_path):
                print(f"Skipping missing hour file: {hour_input_path}")
                continue

            os.makedirs(output_hour_path, exist_ok=True)


            ds = gdal.Open(hour_input_path)
            meta = ds.GetMetadata()

            capture_time_start = meta.get('NC_GLOBAL#time_coverage_start', None)

            timestamp = None

            if not capture_time_start:
                timestamp = get_timestamp(2022, day, hour)
            else:
                timestamp = datetime.strptime(capture_time_start, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()

            for site_id, site in sites.items():
                output_tif_path = os.path.join(output_hour_path, f"{site_id}.tif")

                current_weather = get_site_current_weather(cursor, site_id, timestamp)

                if not current_weather:
                    print(f"Does not have current weather for {site_id}")
                    continue

                predicted = get_site_future_output(cursor, site_id, timestamp)

                if len(predicted) < 6:
                    continue

                srcWin = get_src_win_bounds(ds, site['lat'], site['lon'])

                gdal.Translate(output_tif_path, ds, srcWin=srcWin, format="GTiff")

                global_bounds = get_bound_lat_longs(ds, srcWin)

                if not os.path.exists(output_tif_path):
                    print(f"Failed to create: {output_tif_path}")

                created = gdal.Open(output_tif_path, gdal.GA_Update)


                if not created:
                    os.remove(output_tif_path) # We can't add metadata for whatever reason
                    print(f"{site_id} didn't get opened")
                    continue

                created.SetMetadata({
                    "in_a_timestamp": timestamp,
                    "in_b_ulx": global_bounds[0],
                    "in_c_uly": global_bounds[1],
                    "in_d_lrx": global_bounds[2],
                    "in_e_lry": global_bounds[3],
                    "in_f_hour_of_capture": hour,
                    "in_g_elevation": site["elevation"],
                    "in_h_site_lat": site["lat"],
                    "in_i_site_lon": site["lon"],
                    "in_j_temp": current_weather["temp"],
                    "in_k_wind_speed": current_weather["wind_speed"],
                    "in_l_wind_direction": current_weather["wind_direction"],
                    "in_m_ght": current_weather["ght"],
                    "in_n_wind_speed": current_weather["humidity"],

                    "out_ght_1": predicted[0],
                    "out_ght_2": predicted[1],
                    "out_ght_3": predicted[2],
                    "out_ght_4": predicted[3],
                    "out_ght_5": predicted[4],
                    "out_ght_6": predicted[5]
                })

def get_src_win_bounds(ds, center_lat, center_long, size=128):
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)

    # Convert center lon/lat to pixel
    px, py = gdal.ApplyGeoTransform(inv_gt, center_long, center_lat)
    px = int(round(px))
    py = int(round(py))
    half = size // 2
    xoff = px - half
    yoff = py - half

    return [xoff, yoff, size, size]


def get_bound_lat_longs(ds, srcWin):
    gt = ds.GetGeoTransform()

    # Convert pixel window to geographic bounds
    ulx, uly = gdal.ApplyGeoTransform(gt, srcWin[0], srcWin[1])
    lrx, lry = gdal.ApplyGeoTransform(gt, srcWin[0] + srcWin[2], srcWin[1] + srcWin[3])

    return ulx, uly, lrx, lry  # lon/lat bounds


# def crop_and_get_bounds(input_path, output_path, center_lat, center_lon, size=128):
#     ds = gdal.Open(input_path)
#     gt = ds.GetGeoTransform()
#     inv_gt = gdal.InvGeoTransform(gt)

#     # Convert center lon/lat to pixel
#     px, py = gdal.ApplyGeoTransform(inv_gt, center_lon, center_lat)
#     px = int(round(px))
#     py = int(round(py))
#     half = size // 2
#     xoff = px - half
#     yoff = py - half

#     # Crop
#     gdal.Translate(
#         output_path,
#         ds,
#         srcWin=[xoff, yoff, size, size],
#         format="GTiff"
#     )

#     # Convert pixel window to geographic bounds
#     ulx, uly = gdal.ApplyGeoTransform(gt, xoff, yoff)
#     lrx, lry = gdal.ApplyGeoTransform(gt, xoff + size, yoff + size)

#     return ulx, uly, lrx, lry  # lon/lat bounds


# if __name__ == "__main__":
#     # input_base_path = "/s/parsons/b/others/sustain/diurnalModel/data/TifFolder"
#     # output_base_path = "/s/parsons/b/others/sustain/diurnalModel/quadHash2"

#     chop_to_locations(input_base_path, output_base_path)


def main():
    parser = argparse.ArgumentParser(description="Generate overhead site images with metadata")
    parser.add_argument("database_path", help="Path to the database file")
    parser.add_argument("input_base_path", help="Path to the input files")
    parser.add_argument("output_base_path", help="Path to the output directory")

    args = parser.parse_args()

    chop_to_locations(args.database_path, args.input_base_path, args.output_base_path)

if __name__ == "__main__":
    main()