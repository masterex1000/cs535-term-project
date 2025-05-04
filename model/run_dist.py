import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# CONSTANTS (master does not change)
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

def cleanup():
    dist.destroy_process_group()

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def demo_training(rank, world_size):
    # build model and wrap in DDP
    model     = SimpleNet()
    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    batch_size = 20
    for epoch in range(5):
        # dummy data
        inputs  = torch.randn(batch_size, 10)
        targets = torch.randn(batch_size, 1)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"[Rank {rank}] "
              f"Epoch {epoch} → loss = {loss.item():.4f}",
              flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank",       type=int, help="Rank of this process")
    parser.add_argument("world_size", type=int, help="Total number of processes")
    args = parser.parse_args()

    setup(args.rank, args.world_size)
    demo_training(args.rank, args.world_size)
    cleanup()

if __name__ == "__main__":
    main()
