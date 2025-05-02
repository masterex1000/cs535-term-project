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
from sklearn.metrics import mean_squared_error, r2_score
from dataset import TiffMetadataDataset

# Verify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data augmentation and normalization (single-channel approximation using ImageNet statistics)
transform = T.Compose([
    T.ToPILImage(),                     #  Tensor transfer to PIL Image
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229]),
])

# load dataset
root = "/s/chopin/k/grad/C836699459/cs535-term-project/cs535-term-project/ingest/sat/data/output"
paths = glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)
print(f"Found {len(paths)} .tif files")
if len(paths) == 0:
    raise RuntimeError(f"No .tif files found under {root}")

dataset = TiffMetadataDataset(paths, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoader
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

# model:
class ResNetWithMetadata(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Only train layer4 and fully connected parts
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        # Modify input channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

        # Metadata branch
        self.mlp_meta = nn.Sequential(
            nn.Linear(13, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # BatchNorm after fusion
        self.bn_combined = nn.BatchNorm1d(512 + 64)
        # Final output layer
        self.final = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6),
        )

    def forward(self, img, meta):
        img_feat  = self.resnet(img)
        meta_feat = self.mlp_meta(meta)
        combined  = torch.cat([img_feat, meta_feat], dim=1)
        combined  = self.bn_combined(combined)
        return self.final(combined)

model = ResNetWithMetadata().to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

# Mixed Precision Settings
scaler = GradScaler()

# Training parameters
num_epochs = 50
save_path   = "best_model.pt"
best_val_r2 = -float('inf')
patience    = 10
patience_counter = 0

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for img, meta, label in tqdm(train_loader, desc=f"Train Epoch {epoch}"):  
        img, meta, label = img.to(device), meta.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast():
            preds = model(img, meta)
            loss  = criterion(preds, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    avg_train = running_loss / len(train_loader)

    # verify
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, meta, label in val_loader:
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            preds = model(img, meta)
            val_loss += criterion(preds, label).item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    avg_val = val_loss / len(val_loader)

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Learning rate scheduling
    scheduler.step()

    # check early stop
    if r2 > best_val_r2:
        best_val_r2 = r2
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model: R²={r2:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

print("Training complete.")
