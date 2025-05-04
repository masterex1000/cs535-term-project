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

        original_weights = self.resnet.conv1.weight.data

        # Modify to take grayscale images
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        for param in self.resnet.parameters():
            param.requires_grad = False

        # for name, param in self.resnet.named_parameters():
        #     if 'layer4' in name:
        #     # if 'layer4' in name or 'fc' in name:
        #         param.requires_grad = True

        # self.resnet.conv1.requires_grad_()

        # self.resnet.conv1.requires_grad_()

        # with torch.no_grad():
        #     self.resnet.conv1.weight[:] = original_weights.mean(dim=1, keepdim=True)

        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(self.resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.resnet.fc = nn.Identity()

        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.Linear(32, 32),
        )

        # self.meta_net = nn.Sequential(
        #     nn.Linear(metadata_dim, 64),
        #     nn.ReLU(),
        #     # nn.BatchNorm1d(64),
        #     # nn.Linear(64, 64),
        #     # nn.ReLU(),
        #     # nn.Linear(64, 64),
        #     # nn.ReLU(),
        #     # nn.BatchNorm1d(64),
        #     nn.Linear(64, 64),
        # )

        self.head = nn.Sequential(
            # nn.Linear(512 + 64, 128),
            # nn.Linear(512 + 32, 32),
            nn.Linear(512 + 32, 128),
            # nn.Linear(512, 32),
            # nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim),
            # nn.Linear(32, output_dim),
        )

    def forward(self, image, metadata):
        image_feat = self.resnet(image)
        meta_feat = self.meta_net(metadata)
        x = torch.cat([image_feat, meta_feat], dim=1)
        return self.head(x)
        # return self.head(image_feat)
        # return self.head(meta_feat)