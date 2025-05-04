import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset import TiffMetadataDataset
from torch.nn.parallel import DistributedDataParallel as DDP

# from dataset_rasterio import TiffMetadataDataset
import glob
import tqdm
from solar_resnet18_late import ResNetWithMetadata

MASTER_ADDR = "marlin"
MASTER_PORT = "17171"
BACKEND     = "gloo"

def setup(rank, world_size):
    """
    Initialize the default process group.
    """
    dist.init_process_group(
        backend=BACKEND,
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        world_size=world_size,
        rank=rank
    )
    torch.manual_seed(0)
    

def load_datasets():
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
    
    print(val_dataset.labels.mean(axis=0))
    print(val_dataset.labels.std(axis=0))
    
    return train_dataset, val_dataset

def cleanup():
    dist.destroy_process_group()

# Model

# model = ResNetWithMetadata(metadata_dim=14, output_dim=6).to(device)
# model = ResNetWithMetadata(metadata_dim=len(train_dataset.metadata[0]), output_dim=6).to(device)

# Optimizer & loss
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# optimizer = optim.Adam([
#     {'params': model.resnet.layer4.parameters(), 'lr': 1e-5},
#     {'params': model.meta_net.parameters()},
#     {'params': model.head.parameters()}, 
# ], lr=0.001)


loss_fn = nn.MSELoss()
    
def training(rank, world_size, device, train_dataset, val_dataset):
    model = ResNetWithMetadata(metadata_dim=len(train_dataset.metadata[0]), output_dim=6).to(device)
    ddp_model = DDP(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    batch_size = 32
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=4)
    
    
    for epoch in range(300):
        model.train()
        
        dist.barrier() # Let all processes sync before starting new training epoch
        
        local_train_loss = 0.0
        
        for img, meta, label in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", disable=True):
            img, meta, label = img.to(device), meta.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img, meta)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            local_train_loss += loss.item() * img.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, meta, label in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", disable=True):
                img, meta, label = img.to(device), meta.to(device), label.to(device)
                output = model(img, meta)
                loss = loss_fn(output, label)
                val_loss += loss.item() * img.size(0)


        # Reduce train loss
        loss_tensor = torch.tensor(local_train_loss)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        average_loss = loss_tensor / len(train_dataset)

        # Reduce val loss
        val_tensor = torch.tensor(val_loss)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_tensor / len(val_dataset)

        # print(f"Epoch {epoch+1} Train Loss: {local_train_loss/len(train_dataset):.4f}, Val Loss: {val_loss/len(val_dataset):.4f}")
        print(f"Epoch {epoch+1} Train Loss: {average_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank",       type=int, help="Rank of this process")
    parser.add_argument("world_size", type=int, help="Total number of processes")
    args = parser.parse_args()
    
    
    setup(args.rank, args.world_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device {device}")
    
    train_dataset, val_dataset = load_datasets()
    
    training(args.rank, args.world_size, device, train_dataset, val_dataset)
    
    cleanup()

if __name__ == "__main__":
    main()