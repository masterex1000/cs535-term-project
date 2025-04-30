

import os
import torch
from torch.utils.data import Dataset
from osgeo import gdal
gdal.UseExceptions()

import numpy as np

class SolarDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".tif")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # load image data 
        ds = gdal.Open(image_path)
        img = ds.ReadAsArray().astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Single channel image => [1, H, W]

        # read metadata
        meta = ds.GetMetadata()

        # features（as input）
        features = np.array([
            float(meta.get("temp", 0)),
            float(meta.get("humidity", 0)),
            float(meta.get("wind_speed", 0)),
            float(meta.get("wind_direction", 0)),
            float(meta.get("ght", 0)),
        ], dtype=np.float32)

        # Tags (ght average for the next 6 hours)
        ght_future = [
            float(meta.get(f"ght+{i}", 0)) for i in range(1, 7)
        ]
        label = np.mean(ght_future).astype(np.float32)

        return {
            "image": torch.tensor(img),
            "features": torch.tensor(features),
            "label": torch.tensor(label)
        }
