import glob
from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from osgeo import gdal
import numpy as np
import os

class TiffMetadataDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        transform=None,
        normalize_metadata=True,
        normalize_outputs=True
    ):
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

            try:
                md = ds.GetMetadata()
                metadata = np.array([
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
                ], dtype=np.float32)

                label = np.array([
                    float(md[f'out_ght_{i}']) for i in range(1, 7)
                ], dtype=np.float32)

                self.metadata.append(metadata)
                self.labels.append(label)
            except Exception:
                print(f"Had issues loading {path}. Ignoring")
                continue

        self.metadata = np.stack(self.metadata)
        self.labels = np.stack(self.labels)

        # normalize metadata if desired
        if normalize_metadata:
            self.meta_mean = self.metadata.mean(axis=0)
            self.meta_std = self.metadata.std(axis=0)
            self.metadata = (self.metadata - self.meta_mean) / self.meta_std
        else:
            self.meta_mean = None
            self.meta_std = None

        # normalize outputs if desired
        if normalize_outputs:
            self.label_mean = self.labels.mean(axis=0)
            self.label_std = self.labels.std(axis=0)
            self.labels = (self.labels - self.label_mean) / self.label_std
        else:
            self.label_mean = None
            self.label_std = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        img_array = band.ReadAsArray()

        # cast to float32 *before* creating the tensor, to avoid uint16 errors
        img_tensor = torch.from_numpy(img_array.astype(np.float32))
        # add channel dim and rescale 0–1
        img_tensor = img_tensor.unsqueeze(0) / 65535.0

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        meta_tensor = torch.from_numpy(self.metadata[idx]).float()
        label_tensor = torch.from_numpy(self.labels[idx]).float()

        return img_tensor, meta_tensor, label_tensor
