from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from dataset import TiffMetadataDataset
# from dataset_rasterio import TiffMetadataDataset
import glob
import tqdm
from solar_resnet18_late import ResNetWithMetadata

# Used to create site embedding vector
site_list = ["BMS", "IRRSP", "NWTC", "STAC", "UAT", "ULL", "UOSMRL", "UTPASRL"]

# Load file paths
tiff_files = glob.glob('ingest/sat/data/output/**/**/*.tif')

print(f"Dataset includes {len(tiff_files)} samples")
# train_paths, val_paths = train_test_split(tiff_files, test_size=0.2, random_state=1254)
train_paths, val_paths = train_test_split(tiff_files, test_size=0.1, random_state=1254)

# Transforms
# transform = transforms.Resize((128, 128))  # Already grayscale
transform = None

# Datasets
train_dataset = TiffMetadataDataset(site_list, train_paths, transform=transform)
val_dataset = TiffMetadataDataset(site_list, val_paths, transform=transform,
                                  normalize_metadata=False,
                                  normalize_outputs=False)  # reuse stats

print(f"Loaded datasets ({len(train_dataset.labels)} samples in train, {len(val_dataset.labels)} samples in validate)")

print(f"Input mean: {train_dataset.meta_mean}")
print(f"Input std: {train_dataset.meta_std}")

print(f"Label mean: {train_dataset.label_mean}")
print(f"Label std: {train_dataset.label_std}")
    
# Apply same normalization
val_dataset.meta_mean = train_dataset.meta_mean
val_dataset.meta_std = train_dataset.meta_std
val_dataset.metadata = (val_dataset.metadata - val_dataset.meta_mean) / val_dataset.meta_std

val_dataset.label_mean = train_dataset.label_mean
val_dataset.label_std = train_dataset.label_std
val_dataset.labels = (val_dataset.labels - val_dataset.label_mean) / val_dataset.label_std

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNetWithMetadata(metadata_dim=14, output_dim=6).to(device)
model = ResNetWithMetadata(metadata_dim=len(train_dataset.metadata[0]), output_dim=6).to(device)

# Optimizer & loss
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam([
    {'params': model.resnet.layer4.parameters(), 'lr': 1e-5},
    {'params': model.meta_net.parameters()},
    {'params': model.head.parameters()}, 
], lr=0.001)


loss_fn = nn.MSELoss()

# Training loop
for epoch in range(300):  # adjust as needed
    model.train()
    train_loss = 0.0
    for img, meta, label in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", disable=True):
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img, meta)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.size(0)

    model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for img, meta, label in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
    #         img, meta, label = img.to(device), meta.to(device), label.to(device)
    #         output = model(img, meta)
    #         loss = loss_fn(output, label)
    #         val_loss += loss.item() * img.size(0)

    # print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_dataset):.4f}, Val Loss: {val_loss/len(val_dataset):.4f}")
    
    val_loss = 0.0
    val_mae = 0.0
    val_accuracy = 0.0
    tolerance = 0.1
    
    with torch.no_grad():
        for img, meta, label in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", disable=True):
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            output = model(img, meta)
            loss = loss_fn(output, label)
            val_loss += loss.item() * img.size(0)

            # MAE
            val_mae += torch.sum(torch.abs(output - label)).item()

            # Accuracy within tolerance
            accurate = torch.abs(output - label) < tolerance
            val_accuracy += torch.sum(accurate).item() / label.numel()  # label.numel() = total elements

    # Normalize metrics
    val_loss /= len(val_dataset)
    val_mae /= len(val_dataset) * label.shape[1]  # average per output
    val_accuracy /= len(val_dataset)  # already averaged per batch

    print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_dataset):.4f}, "
        f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Accuracy (<{tolerance}): {val_accuracy:.4f}")