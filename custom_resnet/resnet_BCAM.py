import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch

from attention_module.attention import BCBAM


class Resnet50_BCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.build_layers()
    def build_layers(self):
            resnet_weights = ResNet50_Weights.DEFAULT
            backbone_model = resnet50(weights=resnet_weights)

            self.model_input = nn.Sequential(
                backbone_model.conv1,
                backbone_model.bn1,
                backbone_model.relu,
                backbone_model.maxpool,
            )

            self.layer1 = backbone_model.layer1
            self.layer2 = backbone_model.layer2
            self.layer3 = backbone_model.layer3
            self.BCBAM_layer1 = BCBAM(1024)
            self.layer4 = backbone_model.layer4
            self.BCBAM_layer2 = BCBAM(2048)

            self.avgpool = backbone_model.avgpool
            self.fc = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )

            print("Training on Resnet50 BAM architecture")

    def forward(self, x):
        x = self.model_input(x)
        x = self.layer1(x)
        x= self.layer2(x)
        x = self.layer3(x)
        x = self.BCBAM_layer1(x)

        x = self.layer4(x)

        x = self.BCBAM_layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class Resnet18_BCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.build_layers()
    def build_layers(self):
            resnet_weights = ResNet18_Weights.DEFAULT
            backbone_model = resnet18(weights=resnet_weights)

            self.model_input = nn.Sequential(
                backbone_model.conv1,
                backbone_model.bn1,
                backbone_model.relu,
                backbone_model.maxpool,
            )

            self.layer1 = backbone_model.layer1
            self.layer2 = backbone_model.layer2
            self.layer3 = backbone_model.layer3
            self.BCBAM_layer1 = BCBAM(256)
            self.layer4 = backbone_model.layer4
            self.BCBAM_layer2 = BCBAM(512)

            self.avgpool = backbone_model.avgpool
            self.fc = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, self.num_classes),
                )

            print("Training on Resnet50 BAM architecture")

    def forward(self, x):
        x = self.model_input(x)
        x = self.layer1(x)
        x= self.layer2(x)
        x = self.layer3(x)
        x = self.BCBAM_layer1(x)

        x = self.layer4(x)

        x = self.BCBAM_layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x