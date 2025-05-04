#Train 50K images. 
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







"""
Train Epoch 1: 100%|██████████████████████████| 627/627 [00:10<00:00, 58.12it/s]
Epoch 1 | Train 0.5926 | Val 0.4249 | RMSE 0.6524 | R² 0.5520
Saved best.
Train Epoch 2: 100%|██████████████████████████| 627/627 [00:10<00:00, 60.45it/s]
Epoch 2 | Train 0.4913 | Val 0.3278 | RMSE 0.5731 | R² 0.6540
Saved best.
Train Epoch 3: 100%|██████████████████████████| 627/627 [00:11<00:00, 56.52it/s]
Epoch 3 | Train 0.4372 | Val 0.3700 | RMSE 0.6089 | R² 0.6102
Train Epoch 4: 100%|██████████████████████████| 627/627 [00:10<00:00, 59.97it/s]
Epoch 4 | Train 0.4131 | Val 0.3919 | RMSE 0.6267 | R² 0.5874
Train Epoch 5: 100%|██████████████████████████| 627/627 [00:10<00:00, 60.00it/s]
Epoch 5 | Train 0.3958 | Val 0.3331 | RMSE 0.5777 | R² 0.6488
Train Epoch 6: 100%|██████████████████████████| 627/627 [00:10<00:00, 60.39it/s]
Epoch 6 | Train 0.3691 | Val 0.3162 | RMSE 0.5629 | R² 0.6666
Saved best.
Train Epoch 7: 100%|██████████████████████████| 627/627 [00:10<00:00, 59.70it/s]
Epoch 7 | Train 0.3576 | Val 0.3972 | RMSE 0.6310 | R² 0.5820
Train Epoch 8: 100%|██████████████████████████| 627/627 [00:10<00:00, 60.27it/s]
Epoch 8 | Train 0.3317 | Val 0.3034 | RMSE 0.5514 | R² 0.6801
Saved best.
Train Epoch 9: 100%|██████████████████████████| 627/627 [00:10<00:00, 61.19it/s]
Epoch 9 | Train 0.2937 | Val 0.3920 | RMSE 0.6268 | R² 0.5869
Train Epoch 10: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.04it/s]
Epoch 10 | Train 0.2701 | Val 0.2797 | RMSE 0.5294 | R² 0.7053
Saved best.
Train Epoch 11: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.42it/s]
Epoch 11 | Train 0.2487 | Val 0.3636 | RMSE 0.6036 | R² 0.6176
Train Epoch 12: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.45it/s]
Epoch 12 | Train 0.2292 | Val 0.2941 | RMSE 0.5427 | R² 0.6912
Train Epoch 13: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.33it/s]
Epoch 13 | Train 0.2242 | Val 0.2744 | RMSE 0.5243 | R² 0.7110
Saved best.
Train Epoch 14: 100%|█████████████████████████| 627/627 [00:10<00:00, 58.68it/s]
Epoch 14 | Train 0.2073 | Val 0.2754 | RMSE 0.5254 | R² 0.7105
Train Epoch 15: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.72it/s]
Epoch 15 | Train 0.1917 | Val 0.2474 | RMSE 0.4979 | R² 0.7394
Saved best.
Train Epoch 16: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.51it/s]
Epoch 16 | Train 0.1881 | Val 0.2550 | RMSE 0.5055 | R² 0.7316
Train Epoch 17: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.91it/s]
Epoch 17 | Train 0.1700 | Val 0.2694 | RMSE 0.5196 | R² 0.7165
Train Epoch 18: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.09it/s]
Epoch 18 | Train 0.1690 | Val 0.2583 | RMSE 0.5087 | R² 0.7282
Train Epoch 19: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.57it/s]
Epoch 19 | Train 0.1649 | Val 0.2540 | RMSE 0.5046 | R² 0.7325
Train Epoch 20: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.93it/s]
Epoch 20 | Train 0.1552 | Val 0.2268 | RMSE 0.4768 | R² 0.7610
Saved best.
Train Epoch 21: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.97it/s]
Epoch 21 | Train 0.1583 | Val 0.2517 | RMSE 0.5023 | R² 0.7347
Train Epoch 22: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.69it/s]
Epoch 22 | Train 0.1547 | Val 0.2306 | RMSE 0.4807 | R² 0.7571
Train Epoch 23: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.03it/s]
Epoch 23 | Train 0.1593 | Val 0.2275 | RMSE 0.4775 | R² 0.7604
Train Epoch 24: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.39it/s]
Epoch 24 | Train 0.1519 | Val 0.2375 | RMSE 0.4879 | R² 0.7497
Train Epoch 25: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.03it/s]
Epoch 25 | Train 0.1725 | Val 0.2435 | RMSE 0.4940 | R² 0.7436
Train Epoch 26: 100%|█████████████████████████| 627/627 [00:10<00:00, 58.97it/s]
Epoch 26 | Train 0.1526 | Val 0.2230 | RMSE 0.4728 | R² 0.7650
Saved best.
Train Epoch 27: 100%|█████████████████████████| 627/627 [00:10<00:00, 61.20it/s]
Epoch 27 | Train 0.1562 | Val 0.2935 | RMSE 0.5423 | R² 0.6906
Train Epoch 28: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.64it/s]
Epoch 28 | Train 0.1609 | Val 0.2514 | RMSE 0.5019 | R² 0.7348
Train Epoch 29: 100%|█████████████████████████| 627/627 [00:10<00:00, 61.13it/s]
Epoch 29 | Train 0.1628 | Val 0.2459 | RMSE 0.4964 | R² 0.7410
Train Epoch 30: 100%|█████████████████████████| 627/627 [00:10<00:00, 58.13it/s]
Epoch 30 | Train 0.1529 | Val 0.2086 | RMSE 0.4572 | R² 0.7801
Saved best.
Train Epoch 31: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.92it/s]
Epoch 31 | Train 0.1701 | Val 0.2212 | RMSE 0.4708 | R² 0.7668
Train Epoch 32: 100%|█████████████████████████| 627/627 [00:10<00:00, 58.64it/s]
Epoch 32 | Train 0.1648 | Val 0.2364 | RMSE 0.4866 | R² 0.7508
Train Epoch 33: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.81it/s]
Epoch 33 | Train 0.1602 | Val 0.3035 | RMSE 0.5515 | R² 0.6798
Train Epoch 34: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.85it/s]
Epoch 34 | Train 0.1870 | Val 0.2805 | RMSE 0.5302 | R² 0.7045
Train Epoch 35: 100%|█████████████████████████| 627/627 [00:10<00:00, 58.93it/s]
Epoch 35 | Train 0.1803 | Val 0.2249 | RMSE 0.4747 | R² 0.7629
Train Epoch 36: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.20it/s]
Epoch 36 | Train 0.1818 | Val 0.2644 | RMSE 0.5147 | R² 0.7211
Train Epoch 37: 100%|█████████████████████████| 627/627 [00:10<00:00, 60.07it/s]
Epoch 37 | Train 0.1742 | Val 0.2295 | RMSE 0.4795 | R² 0.7580
Train Epoch 38: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.47it/s]
Epoch 38 | Train 0.1781 | Val 0.2837 | RMSE 0.5332 | R² 0.7008
Train Epoch 39: 100%|█████████████████████████| 627/627 [00:10<00:00, 57.11it/s]
Epoch 39 | Train 0.1698 | Val 0.3070 | RMSE 0.5547 | R² 0.6763
Train Epoch 40: 100%|█████████████████████████| 627/627 [00:10<00:00, 59.91it/s]
Epoch 40 | Train 0.1694 | Val 0.3113 | RMSE 0.5586 | R² 0.6711
Early stopping.
Training complete.
"""


