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
