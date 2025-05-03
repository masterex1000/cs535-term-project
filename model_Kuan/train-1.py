
# train.py
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from dataset import TiffMetadataDataset  # 

# 1. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load Dataset
tif_root = "/Users/mac/Desktop/cs535-term-project/ingest/sat/data/output"
tif_paths = glob.glob(os.path.join(tif_root, "**", "**", "*.tif"), recursive=True)

print(f"Found {len(tif_paths)} .tif files")
if len(tif_paths) == 0:
    raise RuntimeError("No .tif files found! Please check the path: " + tif_root)

dataset = TiffMetadataDataset(tif_paths)

# 3. Split & Dataloaders
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 4. Define Improved Model
class SolarPredictionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.mlp_meta = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(32*4*4 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )

    def forward(self, img, meta):
        img_feat = self.cnn(img)
        meta_feat = self.mlp_meta(meta)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        return self.final(combined)

model = SolarPredictionNet().to(device)

# 5. Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 30
save_path = "best_model.pt"
best_val_r2 = -float('inf')

# 6. Training loop with RMSE and R2
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for img, meta, label in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img, meta)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation with metrics
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img, meta, label in val_loader:
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            output = model(img, meta)
            loss = criterion(output, label)
            val_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    if r2 > best_val_r2:
        best_val_r2 = r2
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model to {save_path} with R²={r2:.4f}")

print("Training complete.")
