import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(__file__))
from solar_dataset import SolarDataset


class SolarNet(nn.Module):
    def __init__(self):
        super(SolarNet, self).__init__()
        # image feture CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # output (B, 32, 1, 1)
        )

        # weather characteristics
        self.weather_fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
        )

        # Comprehensive prediction
        self.fc_final = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Predict the average GHT value for the next 6 hours
        )

    def forward(self, image, weather):
        img_feat = self.cnn(image)
        img_feat = img_feat.view(img_feat.size(0), -1)  # 

        weather_feat = self.weather_fc(weather)

        combined = torch.cat([img_feat, weather_feat], dim=1)
        out = self.fc_final(combined)
        return out.squeeze(1)  # (B,) caculate loss



def train_model():
    # chose device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    dataset = SolarDataset("ingest/sat/data/images")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # initialization
    model = SolarNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            weather = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images, weather)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    # save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/solar_net.pth")
    print("✅ done，model saved： models/solar_net.pth")

if __name__ == "__main__":
    train_model()
