import argparse
from osgeo import gdal

def print_gdal_metadata(filename: str):
    dataset = gdal.Open(filename)
    if not dataset:
        print(f"Failed to open file: {filename}")
        return

    print("Metadata:")
    metadata = dataset.GetMetadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\nProjection:")
    print(dataset.GetProjection())

    print("\nGeoTransform:")
    print(dataset.GetGeoTransform())

def main():
    parser = argparse.ArgumentParser(description="Print GDAL metadata for a .tif file.")
    parser.add_argument("filename", help="Path to the .tif file")
    args = parser.parse_args()

    print_gdal_metadata(args.filename)

if __name__ == "__main__":
    main()