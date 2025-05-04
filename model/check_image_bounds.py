from PIL import Image
import numpy as np
import glob
import tqdm

tiff_files = glob.glob('ingest/sat/data/output/**/**/*.tif')

global_min = float('inf')
global_max = float('-inf')

for path in tqdm.tqdm(tiff_files, desc="Scanning TIFFs"):
    img = Image.open(path)
    arr = np.array(img)
    global_min = min(global_min, arr.min())
    global_max = max(global_max, arr.max())

print(f"Global pixel value range: min = {(global_min / 32767.5) - 1.0}, max = {(global_max/ 32767.5) - 1.0}")