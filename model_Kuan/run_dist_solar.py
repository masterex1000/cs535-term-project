# run_dist_solar.py
import argparse, glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import tqdm
from dataset import TiffMetadataDataset
from torchvision.models import resnet18, ResNet18_Weights

MASTER_ADDR = "marlin"
MASTER_PORT = "17171"
BACKEND = "gloo"

def setup(rank, world_size):
    dist.init_process_group(
        backend=BACKEND,
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        world_size=world_size,
        rank=rank
    )
    torch.manual_seed(0)

def cleanup():
    dist.destroy_process_group()

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
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class ResNetWithCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam = CBAM(512)
        self.avgpool = base.avgpool
        self.mlp_meta = nn.Sequential(
            nn.Linear(13, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        self.bn_combined = nn.BatchNorm1d(512 + 64)
        self.fc = nn.Sequential(
            nn.Linear(576, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, 6)
        )
    def forward(self, img, meta):
        x = self.relu(self.bn1(self.conv1(img)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        m = self.mlp_meta(meta)
        combined = torch.cat([x, m], dim=1)
        combined = self.bn_combined(combined)
        return self.fc(combined)

def load_datasets():
    tiff_files = glob.glob("ingest/sat/data/output/**/**/*.tif", recursive=True)
    dataset = TiffMetadataDataset(tiff_files)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def training(rank, world_size, device, train_dataset, val_dataset):
    model = ResNetWithCBAM().to(device)
    ddp_model = DDP(model, device_ids=None)
    optimizer = optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.layer3.parameters(), 'lr': 5e-5},
        {'params': model.mlp_meta.parameters(), 'lr': 5e-4},
        {'params': model.fc.parameters(), 'lr': 5e-4},
        {'params': model.cbam.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64, num_workers=4)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=64, num_workers=4)

    for epoch in range(1, 51):
        ddp_model.train()
        train_loss = 0.0
        for img, meta, label in tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=(rank != 0)):
            img, meta, label = img.to(device), meta.to(device), label.to(device)
            optimizer.zero_grad()
            with autocast():
                output = ddp_model(img, meta)
                loss = criterion(output, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        dist.barrier()
        scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch}] Training Loss: {train_loss / len(train_loader):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("world_size", type=int)
    args = parser.parse_args()

    setup(args.rank, args.world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = load_datasets()
    training(args.rank, args.world_size, device, train_dataset, val_dataset)
    cleanup()

if __name__ == "__main__":
    main()
