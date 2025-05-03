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
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error, r2_score
from dataset import TiffMetadataDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Data preprocessing
transform = T.Compose([
   T.ToPILImage(),
   T.RandomHorizontalFlip(),
   T.RandomRotation(10),
   T.ToTensor(),
   T.Normalize(mean=[0.485], std=[0.229]),
])


# read path
root = "/s/chopin/k/grad/C836699459/cs535-term-project/cs535-term-project/ingest/sat/data/output"
paths = glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)
print(f"Found {len(paths)} .tif files")
if not paths:
   raise RuntimeError(f"No .tif files under {root}")


# dtatset
dataset = TiffMetadataDataset(
   paths,
   transform=transform,
   normalize_metadata=True,
   normalize_outputs=True,
)


# Division
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(
   dataset, [train_size, val_size],
   generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
   train_ds, batch_size=64, shuffle=True,
   num_workers=8, pin_memory=True, prefetch_factor=4
)
val_loader = DataLoader(
   val_ds, batch_size=64, shuffle=False,
   num_workers=4, pin_memory=True, prefetch_factor=4
)


# CBAM
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
       return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


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


# ResNet + CBAM + metadata
class ResNetWithCBAM(nn.Module):
   def __init__(self):
       super().__init__()
       base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
       # freeze layer1, layer2
       for name,p in base.named_parameters():
           if name.startswith("layer1") or name.startswith("layer2"):
               p.requires_grad = False


       # Change to single channel
       base.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
       self.conv1, self.bn1, self.relu, self.maxpool = (
           base.conv1, base.bn1, base.relu, base.maxpool
       )
       self.layer1, self.layer2 = base.layer1, base.layer2
       self.layer3, self.layer4 = base.layer3, base.layer4
       self.cbam    = CBAM(512)
       self.avgpool = base.avgpool


       # metadata MLP
       self.mlp_meta = nn.Sequential(
           nn.Linear(13,128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1),
           nn.Linear(128,64),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.Dropout(0.1),
       )


       # 576（512+64）**
       ch = 512 + 64
       assert ch == 576, f"combined feature must be 576, got {ch}"
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
       x = self.avgpool(x).flatten(1)  # -> [B,512]
       m = self.mlp_meta(meta)         # -> [B,64]
       comb = torch.cat([x,m], dim=1)  # -> [B,576]
       comb = self.bn_comb(comb)
       return self.fc(comb)


model = ResNetWithCBAM().to(device)


# Optimizer & OneCycleLR
optimizer = torch.optim.AdamW([
   {'params': model.layer4.parameters(),   'lr':1e-4},
   {'params': model.layer3.parameters(),   'lr':5e-5},
   {'params': model.cbam.parameters(),     'lr':1e-4},
   {'params': model.mlp_meta.parameters(), 'lr':5e-4},
   {'params': model.fc.parameters(),       'lr':5e-4},
], weight_decay=1e-4)


EPOCHS = 50
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


save_path, best_r2 = "best_cbam_onecycle.pt", -1e9
patience, cnt = 10, 0


for epoch in range(1, EPOCHS+1):
   model.train(); tr_loss = 0.0
   for img,meta,label in tqdm(train_loader, desc=f"Train Ep {epoch}"):
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
Using device: cuda
Found 66847 .tif files
Train Ep 1: 100%|████████████████████████████████████████████████████████| 836/836 [00:14<00:00, 58.41it/s]
Epoch 1 | Train 0.8904 | Val 0.6694 | RMSE 0.8183 | R² 0.3150
>> Saved new best.
Train Ep 2: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.71it/s]
Epoch 2 | Train 0.6397 | Val 0.6456 | RMSE 0.8036 | R² 0.3368
>> Saved new best.
Train Ep 3: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.54it/s]
Epoch 3 | Train 0.5185 | Val 0.9538 | RMSE 0.9768 | R² 0.0113
Train Ep 4: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.65it/s]
Epoch 4 | Train 0.4618 | Val 0.5045 | RMSE 0.7104 | R² 0.4832
>> Saved new best.
Train Ep 5: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.38it/s]
Epoch 5 | Train 0.4452 | Val 1.4260 | RMSE 1.1944 | R² -0.4806
Train Ep 6: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.40it/s]
Epoch 6 | Train 0.4289 | Val 0.7533 | RMSE 0.8681 | R² 0.2247
Train Ep 7: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.10it/s]
Epoch 7 | Train 0.4024 | Val 0.5286 | RMSE 0.7272 | R² 0.4586
Train Ep 8: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.41it/s]
Epoch 8 | Train 0.3865 | Val 0.6177 | RMSE 0.7861 | R² 0.3632
Train Ep 9: 100%|████████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.30it/s]
Epoch 9 | Train 0.3869 | Val 0.5417 | RMSE 0.7361 | R² 0.4408
Train Ep 10: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.31it/s]
Epoch 10 | Train 0.3644 | Val 0.4126 | RMSE 0.6424 | R² 0.5770
>> Saved new best.
Train Ep 11: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.44it/s]
Epoch 11 | Train 0.3407 | Val 0.4014 | RMSE 0.6336 | R² 0.5890
>> Saved new best.
Train Ep 12: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.20it/s]
Epoch 12 | Train 0.3381 | Val 0.5440 | RMSE 0.7377 | R² 0.4348
Train Ep 13: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.43it/s]
Epoch 13 | Train 0.3201 | Val 0.6586 | RMSE 0.8117 | R² 0.3205
Train Ep 14: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.57it/s]
Epoch 14 | Train 0.3027 | Val 0.5986 | RMSE 0.7739 | R² 0.3798
Train Ep 15: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.54it/s]
Epoch 15 | Train 0.2961 | Val 0.4959 | RMSE 0.7044 | R² 0.4898
Train Ep 16: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.52it/s]
Epoch 16 | Train 0.2584 | Val 0.6316 | RMSE 0.7949 | R² 0.3470
Train Ep 17: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.44it/s]
Epoch 17 | Train 0.2658 | Val 0.5050 | RMSE 0.7108 | R² 0.4766
Train Ep 18: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.44it/s]
Epoch 18 | Train 0.2562 | Val 0.6538 | RMSE 0.8087 | R² 0.3254
Train Ep 19: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.32it/s]
Epoch 19 | Train 0.2308 | Val 0.6334 | RMSE 0.7960 | R² 0.3429
Train Ep 20: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.31it/s]
Epoch 20 | Train 0.2139 | Val 0.5500 | RMSE 0.7418 | R² 0.4356
Train Ep 21: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.23it/s]
Epoch 21 | Train 0.2091 | Val 0.3819 | RMSE 0.6181 | R² 0.6109
>> Saved new best.
Train Ep 22: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.37it/s]
Epoch 22 | Train 0.1949 | Val 0.3166 | RMSE 0.5628 | R² 0.6761
>> Saved new best.
Train Ep 23: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.53it/s]
Epoch 23 | Train 0.1873 | Val 0.3522 | RMSE 0.5936 | R² 0.6393
Train Ep 24: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.33it/s]
Epoch 24 | Train 0.1796 | Val 0.3356 | RMSE 0.5794 | R² 0.6548
Train Ep 25: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.24it/s]
Epoch 25 | Train 0.1678 | Val 0.3214 | RMSE 0.5670 | R² 0.6709
Train Ep 26: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.37it/s]
Epoch 26 | Train 0.1543 | Val 0.3403 | RMSE 0.5834 | R² 0.6504
Train Ep 27: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.22it/s]
Epoch 27 | Train 0.1567 | Val 0.4234 | RMSE 0.6508 | R² 0.5635
Train Ep 28: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.39it/s]
Epoch 28 | Train 0.1530 | Val 0.3129 | RMSE 0.5595 | R² 0.6799
>> Saved new best.
Train Ep 29: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.50it/s]
Epoch 29 | Train 0.1402 | Val 0.3281 | RMSE 0.5729 | R² 0.6635
Train Ep 30: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.47it/s]
Epoch 30 | Train 0.1408 | Val 0.3367 | RMSE 0.5804 | R² 0.6529
Train Ep 31: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.22it/s]
Epoch 31 | Train 0.1338 | Val 0.3366 | RMSE 0.5803 | R² 0.6553
Train Ep 32: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.43it/s]
Epoch 32 | Train 0.1268 | Val 0.3122 | RMSE 0.5588 | R² 0.6803
>> Saved new best.
Train Ep 33: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.44it/s]
Epoch 33 | Train 0.1208 | Val 0.2880 | RMSE 0.5368 | R² 0.7050
>> Saved new best.
Train Ep 34: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.20it/s]
Epoch 34 | Train 0.1168 | Val 0.2864 | RMSE 0.5353 | R² 0.7070
>> Saved new best.
Train Ep 35: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.46it/s]
Epoch 35 | Train 0.1148 | Val 0.3118 | RMSE 0.5585 | R² 0.6803
Train Ep 36: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.47it/s]
Epoch 36 | Train 0.1094 | Val 0.2706 | RMSE 0.5203 | R² 0.7241
>> Saved new best.
Train Ep 37: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.34it/s]
Epoch 37 | Train 0.1050 | Val 0.2857 | RMSE 0.5346 | R² 0.7070
Train Ep 38: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.14it/s]
Epoch 38 | Train 0.1029 | Val 0.2743 | RMSE 0.5238 | R² 0.7189
Train Ep 39: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.39it/s]
Epoch 39 | Train 0.1031 | Val 0.2627 | RMSE 0.5126 | R² 0.7317
>> Saved new best.
Train Ep 40: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.27it/s]
Epoch 40 | Train 0.0999 | Val 0.3026 | RMSE 0.5502 | R² 0.6889
Train Ep 41: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.13it/s]
Epoch 41 | Train 0.0967 | Val 0.2818 | RMSE 0.5309 | R² 0.7127
Train Ep 42: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.41it/s]
Epoch 42 | Train 0.0948 | Val 0.2791 | RMSE 0.5284 | R² 0.7154
Train Ep 43: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.25it/s]
Epoch 43 | Train 0.0940 | Val 0.2620 | RMSE 0.5119 | R² 0.7315
Train Ep 44: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.42it/s]
Epoch 44 | Train 0.0935 | Val 0.2701 | RMSE 0.5198 | R² 0.7239
Train Ep 45: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.25it/s]
Epoch 45 | Train 0.0916 | Val 0.2713 | RMSE 0.5210 | R² 0.7237
Train Ep 46: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.12it/s]
Epoch 46 | Train 0.0919 | Val 0.2796 | RMSE 0.5289 | R² 0.7147
Train Ep 47: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.33it/s]
Epoch 47 | Train 0.0903 | Val 0.2618 | RMSE 0.5118 | R² 0.7330
>> Saved new best.
Train Ep 48: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.12it/s]
Epoch 48 | Train 0.0916 | Val 0.2606 | RMSE 0.5106 | R² 0.7336
>> Saved new best.
Train Ep 49: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.09it/s]
Epoch 49 | Train 0.0886 | Val 0.2622 | RMSE 0.5121 | R² 0.7322
Train Ep 50: 100%|███████████████████████████████████████████████████████| 836/836 [00:12<00:00, 68.38it/s]
Epoch 50 | Train 0.0899 | Val 0.2532 | RMSE 0.5033 | R² 0.7417

"""
