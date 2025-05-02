# train.py
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from dataset import TiffMetadataDataset  # 请

# 1. Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")


# 2. Load Dataset

# use the real .tifpath
tif_root = "/Users/mac/Desktop/cs535-term-project/ingest/sat/data/output"
tif_paths = glob.glob(os.path.join(tif_root, "**", "**", "*.tif"), recursive=True)

print(f"✅ Found {len(tif_paths)} .tif files")
if len(tif_paths) == 0:
    raise RuntimeError("❌ No .tif files found! Please check the path: " + tif_root)

print(f" Found {len(tif_paths)} .tif files.")

dataset = TiffMetadataDataset(tif_paths)


# 3. Split & Dataloaders

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 4. Define Model

class SolarPredictionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.mlp_meta = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(8*8*8 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
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
num_epochs = 20
save_path = "best_model.pt"
best_val_loss = float('inf')

# 6. Training loop

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

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for img, meta, label in val_loader:
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            output = model(img, meta)
            loss = criterion(output, label)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved new best model to {save_path}")

print(" Training complete.")
