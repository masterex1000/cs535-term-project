#Train 50K images. 

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

































import glob
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, r2_score
from dataset import TiffMetadataDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229]),
])

root = "/s/chopin/k/grad/C836699459/cs535-term-project/cs535-term-project/ingest/sat/data/output"
paths = glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)
print(f"Found {len(paths)} .tif files")
if len(paths) == 0:
    raise RuntimeError(f"No .tif files found under {root}")

dataset = TiffMetadataDataset(paths, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)


# Define CBAM module: full channel + spatial attention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# Define ResNet model with CBAM

class ResNetWithCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Freeze the first two layers
        for name, param in base.named_parameters():
            if name.startswith('layer1') or name.startswith('layer2'):
                param.requires_grad = False
        # Change to single channel input
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = base.conv1; self.bn1 = base.bn1; self.relu = base.relu; self.maxpool = base.maxpool
        self.layer1 = base.layer1; self.layer2 = base.layer2; self.layer3 = base.layer3; self.layer4 = base.layer4
        self.cbam = CBAM(512, reduction=16, spatial_kernel=7)
        self.avgpool = base.avgpool
        # Metadata branch
        self.mlp_meta = nn.Sequential(
            nn.Linear(13, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        # Fusion BN and output
        ch = 512 + 64
        self.bn_combined = nn.BatchNorm1d(ch)
        self.fc = nn.Sequential(nn.Linear(ch, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 6))
    def forward(self, img, meta):
        x = self.relu(self.bn1(self.conv1(img)))
        x = self.maxpool(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        m = self.mlp_meta(meta)
        combined = torch.cat([x, m], dim=1)
        combined = self.bn_combined(combined)
        return self.fc(combined)

model = ResNetWithCBAM().to(device)

# Optimizer and Scheduling: Differential LR + CosineAnnealing
optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 5e-5},
    {'params': model.mlp_meta.parameters(), 'lr': 5e-4},
    {'params': model.fc.parameters(),       'lr': 5e-4},
    {'params': model.cbam.parameters(),     'lr': 1e-4},
], weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
criterion = nn.MSELoss()
scaler = GradScaler()

# Training and validation loop: no normalization and weighting (CBAM test only)
save_path = "best_cbam_ablation.pt"
best_r2 = -float('inf')
patience = 10
patience_cnt = 0

for epoch in range(1, 51):
    model.train(); total_loss = 0.0
    for img, meta, label in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        img, meta, label = img.to(device), meta.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast():
            preds = model(img, meta)
            loss = criterion(preds, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    avg_train = total_loss / len(train_loader)

    model.eval(); val_loss = 0.0; preds_list, labels_list = [], []
    with torch.no_grad():
        for img, meta, label in val_loader:
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            output = model(img, meta)
            val_loss += criterion(output, label).item()
            preds_list.append(output.cpu().numpy()); labels_list.append(label.cpu().numpy())
    avg_val = val_loss / len(val_loader)
    y_true = np.concatenate(labels_list, axis=0); y_pred = np.concatenate(preds_list, axis=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)); r2 = r2_score(y_true, y_pred)

    print(f"Epoch {epoch} | Train {avg_train:.4f} | Val {avg_val:.4f} | RMSE {rmse:.4f} | R² {r2:.4f}")
    scheduler.step()

    if r2 > best_r2:
        best_r2 = r2; patience_cnt = 0; torch.save(model.state_dict(), save_path); print("Saved best.")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping."); break
print("Training complete.")
