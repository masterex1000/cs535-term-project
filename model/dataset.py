import glob
from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from osgeo import gdal
import numpy as np
import os

class TiffMetadataDataset(Dataset):
    def __init__(self, file_list: List[str], transform=None, normalize_metadata=True):

        
        self.file_paths = file_list

        self.transform = transform
        self.normalize_metadata = normalize_metadata
        self.metadata = []
        self.labels = []

        # First pass: collect all metadata for normalization
        for path in self.file_paths:
            ds = gdal.Open(path)
            band = ds.GetRasterBand(1)
            img_array = band.ReadAsArray()

            md = ds.GetMetadata()

            print(md)

            metadata = np.array([
                float(md['in_a_timestamp']),
                float(md['in_b_ulx']),
                float(md['in_c_uly']),
                float(md['in_d_lrx']),
                float(md['in_e_lry']),
                float(md['in_f_hour_of_capture']),
                float(md['in_g_elevation']),
                float(md['in_h_site_lat']),
                float(md['in_i_site_lon']),
                float(md['in_j_temp']),
                float(md['in_k_wind_speed']),
                float(md['in_l_wind_direction']),
                float(md['in_m_ght']),
                float(md['in_n_wind_speed']),
            ])


            label = np.array([
                float(md[f'out_ght_{i}']) for i in range(1, 7) # 1 through 6 inclusive
            ])

            self.metadata.append(metadata)
            self.labels.append(label)

        self.metadata = np.stack(self.metadata)
        self.labels = np.stack(self.labels)

        if normalize_metadata:
            self.meta_mean = self.metadata.mean(axis=0)
            self.meta_std = self.metadata.std(axis=0)
            self.metadata = (self.metadata - self.meta_mean) / self.meta_std
        else:
            self.meta_mean = None
            self.meta_std = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        ds = gdal.Open(path)
        img_array = ds.GetRasterBand(1).ReadAsArray()

        img = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        if self.transform:
            img = self.transform(img)

        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, metadata, label