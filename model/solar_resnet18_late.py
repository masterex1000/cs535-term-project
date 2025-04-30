import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithMetadata(nn.Module):
    def __init__(self, metadata_dim, output_dim=6):
        super().__init__()
        # self.resnet = models.resnet18(pretrained=True)
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify to take grayscale images
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = False

        # with torch.no_grad():
        #     self.resnet.conv1.weight = nn.Parameter(self.resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.resnet.fc = nn.Identity()

        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

    def forward(self, image, metadata):
        image_feat = self.resnet(image)
        meta_feat = self.meta_net(metadata)
        x = torch.cat([image_feat, meta_feat], dim=1)
        return self.head(x)