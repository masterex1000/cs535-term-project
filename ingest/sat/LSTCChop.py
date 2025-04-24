import os
import socket
import geopandas as gpd
import numpy as np
from osgeo import gdal


def chop_in_quadhash(input_base_path, output_base_path):

    root_path = "/s/parsons/b/others/sustain/diurnalModel/states_quads(9)"

    quadhash_dir = next(
        d
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d)) and d.startswith("quadshape_")
    )

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, "quadhash.shp"))

    # in_path = (
    #     "/s/"
    #     + socket.gethostname()
    #     + "/a/all/all/all/sustain/varsh/Python/GOES/TifFolder/001/00.tif"
    # )
    # out_path = (
    #     "/s/"
    #     + socket.gethostname()
    #     + "/a/all/all/all/sustain/varsh/Python/GOES/quadHash/"
    # )

    for day in range(365, 366):
        day_dir = f"{day:03d}"
        day_input_path = os.path.join(input_base_path, day_dir)
        output_day_path = os.path.join(output_base_path, day_dir)

        if not os.path.exists(day_input_path):
            print(f"Skipping missing day directory: {day_input_path}")
            continue

        os.makedirs(output_day_path, exist_ok=True)

        for hour in range(24):  # 00 to 23 hours
            hour_dir = f"{hour:02d}"
            hour_input_path = os.path.join(day_input_path, f"{hour_dir}.tif")
            output_hour_path = os.path.join(output_day_path, hour_dir)

            if not os.path.exists(hour_input_path):
                print(f"Skipping missing hour file: {hour_input_path}")
                continue

            os.makedirs(output_hour_path, exist_ok=True)

            count = 0
            total = len(quadhashes)

            for ind, row in quadhashes.iterrows():
                poly, quadkey = row["geometry"], row["Quadkey"]

                count += 1
                print(f"Splitting: Day {day}, Hour {hour}, {count} / {total}")

                quadkey_folder = os.path.join(output_hour_path, quadkey)
                os.makedirs(quadkey_folder, exist_ok=True)

                bounds = list(poly.exterior.coords)
                window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

                output_tif_path = os.path.join(quadkey_folder, "nlcd.tif")
                gdal.Translate(output_tif_path, hour_input_path, projWin=window)

                if os.path.exists(output_tif_path):
                    x = gdal.Open(output_tif_path).ReadAsArray()
                    if np.min(x) == np.max(x) == 0:
                        os.remove(output_tif_path)  # Remove empty TIFF
                        print(f"Removed empty file: {output_tif_path}")
                    else:
                        print(f"Created: {output_tif_path}")
                else:
                    print(f"Failed to create: {output_tif_path}")

    # for ind, row in quadhashes.iterrows():
    #     poly, quadkey = row["geometry"], row["Quadkey"]

    #     count += 1
    #     print("Splitting: ", count, "/", total)

    #     os.makedirs(os.path.join(out_path, quadkey), exist_ok=True)

    #     bounds = list(poly.exterior.coords)
    #     window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

    #     output_tif_path = os.path.join(out_path, quadkey, "nlcd.tif")

    #     gdal.Translate(output_tif_path, in_path, projWin=window)

    #     if os.path.exists(output_tif_path):
    #         x = gdal.Open(output_tif_path).ReadAsArray()
    #         if np.min(x) == np.max(x) == 0:
    #             os.remove(output_tif_path)  # Remove empty TIFF
    #             print(f"Removed empty file: {output_tif_path}")
    #         else:
    #             print(f"Created: {output_tif_path}")
    #     else:
    #         print(f"Failed to create: {output_tif_path}")


# if __name__ == "__main__":
#     chop_in_quadhash()

if __name__ == "__main__":
    input_base_path = (
        "/s/parsons/b/others/sustain/diurnalModel/data/TifFolder"
    )
    output_base_path = (
        "/s/parsons/b/others/sustain/diurnalModel/quadHash2"
    )

    chop_in_quadhash(input_base_path, output_base_path)
