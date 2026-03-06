import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch

from attention_module.attention import BidirectionalAttentionModule

class Resnet50_BAM(nn.Module):
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
            self.BAM_layer1 = BidirectionalAttentionModule(1024)
            self.layer4 = backbone_model.layer4
            self.BAM_layer2 = BidirectionalAttentionModule(2048)

            self.avgpool = backbone_model.avgpool
            self.fc = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )

            print("Training on Resnet50 BAM architecture")
    def augment_feature(self, x):
        if self.training: #Biến này kế thừa
            noise = 0.01 * torch.randn_like(x)
            return x + noise

        return x
    def forward(self, x):
        x = self.model_input(x)
        x = self.layer1(x)
        x= self.layer2(x)
        x = self.layer3(x)
        #BAM
        x_aug = self.augment_feature(x)
        x = x + self.BAM_layer1(x, x_aug)

        x = self.layer4(x)

        x_aug = self.augment_feature(x)
        x = x + self.BAM_layer2(x, x_aug)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x