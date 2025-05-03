#100 EPOCHS

import glob
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.models import resnet34, ResNet34_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error, r2_score
from dataset import TiffMetadataDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Change to single channel Normalize
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
if not paths:
    raise RuntimeError(f"No .tif files under {root}")

dataset = TiffMetadataDataset(
    paths,
    transform=transform,
    normalize_metadata=True,
    normalize_outputs=True,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=8, pin_memory=True, prefetch_factor=4)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True, prefetch_factor=4)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//reduction, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) +
                            self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = 3 if kernel_size==7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size,padding=pad,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1,keepdim=True)
        mx, _ = x.max(dim=1,keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg,mx],dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x):
        return x * self.sa(x * self.ca(x))

class ResNetWithCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        for name,p in base.named_parameters():
            if name.startswith("layer1") or name.startswith("layer2"):
                p.requires_grad = False

        base.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.conv1, self.bn1, self.relu, self.maxpool = (
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.cbam    = CBAM(512)
        self.avgpool = base.avgpool

        self.mlp_meta = nn.Sequential(
            nn.Linear(13,128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128,64),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.Dropout(0.1),
        )

        ch = 512 + 64
        self.bn_comb = nn.BatchNorm1d(ch)
        self.fc      = nn.Sequential(
            nn.Linear(ch,128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128,6)
        )

    def forward(self, img, meta):
        x = self.relu(self.bn1(self.conv1(img)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        m = self.mlp_meta(meta)
        comb = torch.cat([x,m], dim=1)
        comb = self.bn_comb(comb)
        return self.fc(comb)

model = ResNetWithCBAM().to(device)

optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(),   'lr':1e-4},
    {'params': model.layer3.parameters(),   'lr':5e-5},
    {'params': model.cbam.parameters(),     'lr':1e-4},
    {'params': model.mlp_meta.parameters(), 'lr':5e-4},
    {'params': model.fc.parameters(),       'lr':5e-4},
], weight_decay=1e-4)

EPOCHS = 100
scheduler = OneCycleLR(
    optimizer,
    max_lr=[1e-4,5e-5,1e-4,5e-4,5e-4],
    total_steps=EPOCHS * len(train_loader),
    pct_start=0.2,
    anneal_strategy='cos',
    cycle_momentum=False,
)

criterion = nn.MSELoss()
scaler    = GradScaler()

save_path, best_r2 = "best_cbam_onecycle34.pt", -1e9
patience, cnt = 10, 0

for epoch in range(1, EPOCHS+1):
    model.train(); tr_loss = 0.0
    for img,meta,label in tqdm(train_loader, desc=f"Train Ep{epoch}"):
        img,meta,label = img.to(device), meta.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(img, meta)
            loss = criterion(out, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        scheduler.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    model.eval(); va_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for img,meta,label in val_loader:
            img,meta,label = img.to(device), meta.to(device), label.to(device)
            out = model(img, meta)
            va_loss += criterion(out, label).item()
            preds.append(out.cpu().numpy()); trues.append(label.cpu().numpy())
    va_loss /= len(val_loader)
    y_true = np.concatenate(trues, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"Epoch {epoch} | Train {tr_loss:.4f} | Val {va_loss:.4f} | RMSE {rmse:.4f} | R² {r2:.4f}")
    if r2 > best_r2:
        best_r2, cnt = r2, 0
        torch.save(model.state_dict(), save_path)
        print(">> Saved new best.")
    else:
        cnt += 1
        if cnt >= patience:
            print(">> Early stopping."); break

print("Training complete.")

"""
grouper:~/cs535-term-project/cs535-term-project$ python model_Kuan/train_resnet18.py 
Using device: cuda
Found 66847 .tif files
Train Ep1: 100%|█████████████████████████████████████████████████████████| 836/836 [00:20<00:00, 41.60it/s]
Epoch 1 | Train 0.8769 | Val 0.6675 | RMSE 0.8171 | R² 0.3173
>> Saved new best.
Train Ep2: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 47.16it/s]
Epoch 2 | Train 0.6793 | Val 0.5438 | RMSE 0.7375 | R² 0.4459
>> Saved new best.
Train Ep3: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 47.00it/s]
Epoch 3 | Train 0.5668 | Val 0.6508 | RMSE 0.8069 | R² 0.3313
Train Ep4: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.88it/s]
Epoch 4 | Train 0.5123 | Val 0.7278 | RMSE 0.8533 | R² 0.2499
Train Ep5: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.76it/s]
Epoch 5 | Train 0.4724 | Val 0.5149 | RMSE 0.7177 | R² 0.4738
>> Saved new best.
Train Ep6: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.78it/s]
Epoch 6 | Train 0.4423 | Val 0.4392 | RMSE 0.6629 | R² 0.5518
>> Saved new best.
Train Ep7: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.72it/s]
Epoch 7 | Train 0.4351 | Val 0.5186 | RMSE 0.7203 | R² 0.4682
Train Ep8: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.71it/s]
Epoch 8 | Train 0.4031 | Val 0.5208 | RMSE 0.7218 | R² 0.4651
Train Ep9: 100%|█████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.77it/s]
Epoch 9 | Train 0.4042 | Val 0.5017 | RMSE 0.7084 | R² 0.4842
Train Ep10: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.69it/s]
Epoch 10 | Train 0.3864 | Val 0.4300 | RMSE 0.6559 | R² 0.5607
>> Saved new best.
Train Ep11: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 11 | Train 0.3676 | Val 0.4480 | RMSE 0.6695 | R² 0.5426
Train Ep12: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.69it/s]
Epoch 12 | Train 0.3678 | Val 0.6346 | RMSE 0.7967 | R² 0.3456
Train Ep13: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.67it/s]
Epoch 13 | Train 0.3577 | Val 0.5540 | RMSE 0.7445 | R² 0.4305
Train Ep14: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
Epoch 14 | Train 0.3531 | Val 0.4858 | RMSE 0.6971 | R² 0.5011
Train Ep15: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.74it/s]
Epoch 15 | Train 0.3422 | Val 0.4603 | RMSE 0.6786 | R² 0.5265
Train Ep16: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.60it/s]
Epoch 16 | Train 0.3356 | Val 0.6391 | RMSE 0.7996 | R² 0.3373
Train Ep17: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.67it/s]
Epoch 17 | Train 0.3347 | Val 0.4735 | RMSE 0.6882 | R² 0.5090
Train Ep18: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.60it/s]
Epoch 18 | Train 0.3076 | Val 0.4131 | RMSE 0.6428 | R² 0.5767
>> Saved new best.
Train Ep19: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.66it/s]
Epoch 19 | Train 0.3164 | Val 0.4924 | RMSE 0.7018 | R² 0.4950
Train Ep20: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 20 | Train 0.3035 | Val 0.3883 | RMSE 0.6232 | R² 0.6015
>> Saved new best.
Train Ep21: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 21 | Train 0.2997 | Val 0.3841 | RMSE 0.6197 | R² 0.6094
>> Saved new best.
Train Ep22: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 22 | Train 0.2914 | Val 0.4316 | RMSE 0.6571 | R² 0.5581
Train Ep23: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.61it/s]
Epoch 23 | Train 0.2763 | Val 0.3732 | RMSE 0.6110 | R² 0.6202
>> Saved new best.
Train Ep24: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.57it/s]
Epoch 24 | Train 0.2587 | Val 0.4389 | RMSE 0.6626 | R² 0.5498
Train Ep25: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.62it/s]
Epoch 25 | Train 0.2463 | Val 0.4109 | RMSE 0.6411 | R² 0.5784
Train Ep26: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.77it/s]
Epoch 26 | Train 0.2374 | Val 0.3304 | RMSE 0.5749 | R² 0.6628
>> Saved new best.
Train Ep27: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.67it/s]
Epoch 27 | Train 0.2197 | Val 0.4638 | RMSE 0.6812 | R² 0.5239
Train Ep28: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.73it/s]
Epoch 28 | Train 0.2279 | Val 0.3841 | RMSE 0.6199 | R² 0.6084
Train Ep29: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.70it/s]
Epoch 29 | Train 0.2184 | Val 0.3739 | RMSE 0.6116 | R² 0.6178
Train Ep30: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
Epoch 30 | Train 0.2058 | Val 0.3209 | RMSE 0.5666 | R² 0.6733
>> Saved new best.
Train Ep31: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.70it/s]
Epoch 31 | Train 0.1964 | Val 0.3217 | RMSE 0.5672 | R² 0.6715
Train Ep32: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.66it/s]
Epoch 32 | Train 0.2068 | Val 0.4776 | RMSE 0.6912 | R² 0.5096
Train Ep33: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.75it/s]
Epoch 33 | Train 0.1716 | Val 0.3542 | RMSE 0.5952 | R² 0.6380
Train Ep34: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 34 | Train 0.1740 | Val 0.3116 | RMSE 0.5583 | R² 0.6819
>> Saved new best.
Train Ep35: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.71it/s]
Epoch 35 | Train 0.1871 | Val 0.3753 | RMSE 0.6127 | R² 0.6148
Train Ep36: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 36 | Train 0.1665 | Val 0.3055 | RMSE 0.5528 | R² 0.6880
>> Saved new best.
Train Ep37: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.73it/s]
Epoch 37 | Train 0.1582 | Val 0.3376 | RMSE 0.5812 | R² 0.6551
Train Ep38: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.63it/s]
Epoch 38 | Train 0.1582 | Val 0.2972 | RMSE 0.5453 | R² 0.6972
>> Saved new best.
Train Ep39: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
Epoch 39 | Train 0.1487 | Val 0.2944 | RMSE 0.5426 | R² 0.7007
>> Saved new best.
Train Ep40: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.73it/s]
Epoch 40 | Train 0.1546 | Val 0.2870 | RMSE 0.5358 | R² 0.7074
>> Saved new best.
Train Ep41: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 41 | Train 0.1551 | Val 0.2836 | RMSE 0.5326 | R² 0.7112
>> Saved new best.
Train Ep42: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.70it/s]
Epoch 42 | Train 0.1363 | Val 0.3093 | RMSE 0.5563 | R² 0.6851
Train Ep43: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.71it/s]
Epoch 43 | Train 0.1358 | Val 0.2914 | RMSE 0.5399 | R² 0.7033
Train Ep44: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.77it/s]
Epoch 44 | Train 0.1340 | Val 0.2821 | RMSE 0.5312 | R² 0.7121
>> Saved new best.
Train Ep45: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.71it/s]
Epoch 45 | Train 0.1296 | Val 0.2856 | RMSE 0.5345 | R² 0.7093
Train Ep46: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
Epoch 46 | Train 0.1325 | Val 0.3159 | RMSE 0.5621 | R² 0.6770
Train Ep47: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
Epoch 47 | Train 0.1278 | Val 0.2793 | RMSE 0.5286 | R² 0.7152
>> Saved new best.
Train Ep48: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.61it/s]
Epoch 48 | Train 0.1144 | Val 0.2800 | RMSE 0.5293 | R² 0.7145
Train Ep49: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.63it/s]
Epoch 49 | Train 0.1131 | Val 0.2927 | RMSE 0.5412 | R² 0.7009
Train Ep50: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.60it/s]
Epoch 50 | Train 0.1153 | Val 0.3189 | RMSE 0.5648 | R² 0.6749
Train Ep51: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.61it/s]
Epoch 51 | Train 0.1159 | Val 0.2844 | RMSE 0.5334 | R² 0.7103
Train Ep52: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.58it/s]
Epoch 52 | Train 0.1119 | Val 0.3099 | RMSE 0.5568 | R² 0.6848
Train Ep53: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 53 | Train 0.1052 | Val 0.2705 | RMSE 0.5202 | R² 0.7245
>> Saved new best.
Train Ep54: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.61it/s]
Epoch 54 | Train 0.1036 | Val 0.2903 | RMSE 0.5389 | R² 0.7049
Train Ep55: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.70it/s]
Epoch 55 | Train 0.1019 | Val 0.2801 | RMSE 0.5294 | R² 0.7139
Train Ep56: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
Epoch 56 | Train 0.0998 | Val 0.2855 | RMSE 0.5344 | R² 0.7093
Train Ep57: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.61it/s]
Epoch 57 | Train 0.0955 | Val 0.2663 | RMSE 0.5162 | R² 0.7285
>> Saved new best.
Train Ep58: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.61it/s]
Epoch 58 | Train 0.0935 | Val 0.2791 | RMSE 0.5284 | R² 0.7157
Train Ep59: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 59 | Train 0.0945 | Val 0.2700 | RMSE 0.5197 | R² 0.7246
Train Ep60: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.73it/s]
Epoch 60 | Train 0.0916 | Val 0.2712 | RMSE 0.5209 | R² 0.7236
Train Ep61: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 61 | Train 0.0902 | Val 0.2763 | RMSE 0.5257 | R² 0.7185
Train Ep62: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.62it/s]
Epoch 62 | Train 0.0849 | Val 0.2834 | RMSE 0.5324 | R² 0.7111
Train Ep63: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.74it/s]
Epoch 63 | Train 0.0876 | Val 0.2660 | RMSE 0.5159 | R² 0.7286
>> Saved new best.
Train Ep64: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.67it/s]
Epoch 64 | Train 0.0847 | Val 0.2729 | RMSE 0.5225 | R² 0.7219
Train Ep65: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.70it/s]
Epoch 65 | Train 0.0817 | Val 0.2601 | RMSE 0.5101 | R² 0.7343
>> Saved new best.
Train Ep66: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.72it/s]
Epoch 66 | Train 0.0780 | Val 0.2627 | RMSE 0.5126 | R² 0.7318
Train Ep67: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 67 | Train 0.0781 | Val 0.2639 | RMSE 0.5138 | R² 0.7313
Train Ep68: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.63it/s]
Epoch 68 | Train 0.0739 | Val 0.3318 | RMSE 0.5761 | R² 0.6614
Train Ep69: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.63it/s]
Epoch 69 | Train 0.0731 | Val 0.2722 | RMSE 0.5218 | R² 0.7225
Train Ep70: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.71it/s]
Epoch 70 | Train 0.0729 | Val 0.2677 | RMSE 0.5175 | R² 0.7275
Train Ep71: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.62it/s]
Epoch 71 | Train 0.0715 | Val 0.2774 | RMSE 0.5268 | R² 0.7169
Train Ep72: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.73it/s]
Epoch 72 | Train 0.0695 | Val 0.2750 | RMSE 0.5245 | R² 0.7195
Train Ep73: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 73 | Train 0.0685 | Val 0.2612 | RMSE 0.5112 | R² 0.7338
Train Ep74: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.69it/s]
Epoch 74 | Train 0.0675 | Val 0.2583 | RMSE 0.5083 | R² 0.7368
>> Saved new best.
Train Ep75: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.66it/s]
Epoch 75 | Train 0.0666 | Val 0.2585 | RMSE 0.5086 | R² 0.7364
Train Ep76: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 76 | Train 0.0665 | Val 0.2615 | RMSE 0.5114 | R² 0.7333
Train Ep77: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.70it/s]
Epoch 77 | Train 0.0641 | Val 0.2569 | RMSE 0.5069 | R² 0.7381
>> Saved new best.
Train Ep78: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.67it/s]
Epoch 78 | Train 0.0631 | Val 0.2640 | RMSE 0.5139 | R² 0.7309
Train Ep79: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 79 | Train 0.0613 | Val 0.2487 | RMSE 0.4988 | R² 0.7463
>> Saved new best.
Train Ep80: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.66it/s]
Epoch 80 | Train 0.0610 | Val 0.2593 | RMSE 0.5093 | R² 0.7357
Train Ep81: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.63it/s]
Epoch 81 | Train 0.0591 | Val 0.2774 | RMSE 0.5268 | R² 0.7172
Train Ep82: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 82 | Train 0.0582 | Val 0.2536 | RMSE 0.5037 | R² 0.7417
Train Ep83: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.66it/s]
Epoch 83 | Train 0.0572 | Val 0.2604 | RMSE 0.5104 | R² 0.7346
Train Ep84: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.59it/s]
Epoch 84 | Train 0.0575 | Val 0.2837 | RMSE 0.5327 | R² 0.7107
Train Ep85: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.63it/s]
Epoch 85 | Train 0.0563 | Val 0.2663 | RMSE 0.5161 | R² 0.7287
Train Ep86: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.66it/s]
Epoch 86 | Train 0.0569 | Val 0.2650 | RMSE 0.5149 | R² 0.7299
Train Ep87: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.64it/s]
Epoch 87 | Train 0.0565 | Val 0.2678 | RMSE 0.5176 | R² 0.7272
Train Ep88: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.68it/s]
Epoch 88 | Train 0.0538 | Val 0.2614 | RMSE 0.5114 | R² 0.7338
Train Ep89: 100%|████████████████████████████████████████████████████████| 836/836 [00:17<00:00, 46.65it/s]
^[Epoch 89 | Train 0.0542 | Val 0.2588 | RMSE 0.5089 | R² 0.7362
>> Early stopping.
Training complete.

"""

