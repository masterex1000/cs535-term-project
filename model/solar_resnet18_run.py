from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from dataset import TiffMetadataDataset
import glob
import tqdm
from solar_resnet18_late import ResNetWithMetadata

# Load file paths
# tiff_files = glob.glob('../ingest/sat/data/output/**/**/*.tif')  # <- Change this
# tiff_files = glob.glob('../ingest/sat/data/output/001/**/*.tif')  # <- Change this
tiff_files = glob.glob('../ingest/sat/data/output/01*/**/*.tif')  # <- Change this


print(len(tiff_files))
train_paths, val_paths = train_test_split(tiff_files, test_size=0.2, random_state=42)

# Transforms
transform = transforms.Resize((128, 128))  # Already grayscale

# Datasets
train_dataset = TiffMetadataDataset(train_paths, transform=transform)
val_dataset = TiffMetadataDataset(val_paths, transform=transform,
                                  normalize_metadata=False)  # reuse stats

# Apply same normalization
val_dataset.meta_mean = train_dataset.meta_mean
val_dataset.meta_std = train_dataset.meta_std
val_dataset.metadata = (val_dataset.metadata - val_dataset.meta_mean) / val_dataset.meta_std

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetWithMetadata(metadata_dim=14, output_dim=6).to(device)

# Optimizer & loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(10):  # adjust as needed
    model.train()
    train_loss = 0.0
    for img, meta, label in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img, meta)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img, meta, label in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            output = model(img, meta)
            loss = loss_fn(output, label)
            val_loss += loss.item() * img.size(0)

    print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_dataset):.4f}, Val Loss: {val_loss/len(val_dataset):.4f}")