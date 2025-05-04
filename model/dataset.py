import glob
import math
from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from osgeo import gdal
import numpy as np
import os

class TiffMetadataDataset(Dataset):
    def __init__(self, site_list: List[str], file_list: List[str], transform=None, normalize_metadata=True, normalize_outputs=True):
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

                # print(md)

                wind_direction = float(md['in_l_wind_direction'])

                wind_dir_rad = np.deg2rad(wind_direction)

                wind_dir_input = (np.cos(wind_dir_rad), np.sin(wind_dir_rad))
                
                site_id = os.path.splitext(os.path.basename(path))[0]

                site_encoding_vector = [1 if site_id == s else -1 for s in site_list]
                
                solar_minutes, solar_angle_rad = solar_time_from_timestamp(float(md['in_p_timestamp_mid']), float(md['in_i_site_lon']))
                
                metadata = np.array([
                    # float(md['in_a_timestamp']),
                    float(md['in_p_timestamp_mid']), # Use midpoint timestamp
                    # float(md['in_b_ulx']),
                    # float(md['in_c_uly']),
                    # float(md['in_d_lrx']),
                    # float(md['in_e_lry']),
                    float(md['in_f_hour_of_capture']),
                    float(md['in_g_elevation']),
                    # float(md['in_h_site_lat']),
                    # float(md['in_i_site_lon']),
                    float(md['in_j_temp']),
                    float(md['in_k_wind_speed']),
                    wind_dir_input[0],
                    wind_dir_input[1],
                    
                    solar_minutes,
                    np.cos(solar_angle_rad),
                    np.sin(solar_angle_rad),
                    
                    # float(md['in_l_wind_direction']),
                    *site_encoding_vector,
                    float(md['in_m_ght']),
                    float(md['in_n_wind_speed']),
                    
                    float(md['in_o_prev_ght_1'])
                ])


                label = np.array([
                    float(md[f'out_ght_{i}']) for i in range(1, 7) # 1 through 6 inclusive
                ])

                self.metadata.append(metadata)
                self.labels.append(label)
                
            except:
                print(f"Had issues loading {path}. Ignoring")
                pass

        self.metadata = np.stack(self.metadata)
        self.labels = np.stack(self.labels)

        if normalize_metadata:
            self.meta_mean = self.metadata.mean(axis=0)
            self.meta_std = self.metadata.std(axis=0)
            self.metadata = (self.metadata - self.meta_mean) / self.meta_std
        else:
            self.meta_mean = None
            self.meta_std = None
            
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
        img_array = ds.GetRasterBand(1).ReadAsArray()

        # img = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        # img = (torch.tensor(img_array.astype(np.float32)).unsqueeze(0) / 65535.0)



        # img = torch.tensor(img_array.astype(np.float32)).unsqueeze(0)
        # img = (img / 32767.5) - 1.0 # Normalize


        # According to https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
        # Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

        # So....

        # Scale to 0.0 to 1.0
        img = (torch.tensor(img_array.astype(np.float32)).unsqueeze(0) / 65535.0)

        # Normalize using mean (average of all three is 0.449) and std (average of three is 0.226)
        # That gives a range of ~(-2..2)

        img = (img - 0.449) / 0.226

        if self.transform:
            img = self.transform(img)

        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, metadata, label

def solar_time_from_timestamp(timestamp, longitude_deg):
    SECONDS_PER_DAY = 86400

    # Days since epoch (1970-01-01), including fraction
    days_since_epoch = timestamp / SECONDS_PER_DAY
    day_of_year = int((days_since_epoch % 365.25)) + 1  # Roughly OK for most NN use

    # Equation of Time (in minutes)
    B = 2 * math.pi * (day_of_year - 81) / 364
    EoT = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    # UTC time of day in minutes
    utc_minutes = (timestamp % SECONDS_PER_DAY) / 60

    # Solar time = UTC + offset from Equation of Time and longitude
    solar_minutes = utc_minutes + EoT + 4 * longitude_deg
    solar_minutes %= 1440  # Wrap around 24h

    # Convert to solar angle (0–360 degrees)
    sun_angle_deg = (solar_minutes / 1440) * 360
    sun_angle_rad = math.radians(sun_angle_deg)
    
    return solar_minutes, sun_angle_rad