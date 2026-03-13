import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, swin_v2_b, Swin_V2_B_Weights
import torch
import torch.nn.functional as F

class CNNtoSwinAdapter(nn.Module):
    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = x.permute(0, 2, 3, 1)
        return x

class Resnet50_Swin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.build_layers()
    def build_layers(self):
            resnet_weights = ResNet50_Weights.DEFAULT
            backbone_model = resnet50(weights=resnet_weights)

            swin_weights = Swin_V2_B_Weights.DEFAULT
            swin_model = swin_v2_b(weights=swin_weights)

            self.model_input = nn.Sequential(
                backbone_model.conv1,
                backbone_model.bn1,
                backbone_model.relu,
                backbone_model.maxpool,
            )
            #Resnet 50
            self.layer1 = backbone_model.layer1
            self.layer2 = backbone_model.layer2
            self.layer3 = backbone_model.layer3
            self.layer4 = backbone_model.layer4
            

            #Swin
            self.swin_layer = swin_model.features[7]

            self.adapt_cnn_2_Swin = CNNtoSwinAdapter() #Dùng để đổi [B, 16, 16, 1024] sang [B, 1024, 8, 8]
            self.swin_proj = nn.Conv2d(1024, 2048, 1)
            
            self.fusion = nn.Conv2d(4096, 2048, kernel_size=1)
            self.avgpool = backbone_model.avgpool
            self.fc = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )

            print("Training on Resnet50 Swin architecture")

    def forward(self, x):
        x = self.model_input(x)
        x = self.layer1(x)
        x= self.layer2(x)
        #Layer này chia nhánh ra 2 nhánh, nhánh 1 vào layer4 gốc của Resnet và nhánh 2 vào stage 3 và 4 của Swin
        shared = self.layer3(x) #(1024, 16, 16)

        #Branch 1
        resnet_branch = self.layer4(shared) #(2048, 8, 8)
        
        #Branch 2
        swin_branch = self.adapt_cnn_2_Swin(shared)  # BCHW -> BHWC
        swin_branch = self.swin_layer(swin_branch)
        swin_branch = swin_branch.permute(0, 3, 1, 2).contiguous()
        swin_branch = self.swin_proj(swin_branch)
        x = torch.cat([resnet_branch, swin_branch], dim=1) 
        x = self.fusion(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class Resnet18_CBAM(nn.Module):
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
            self.CBAM_layer1 = CBAM(256)
            self.layer4 = backbone_model.layer4
            self.CBAM_layer2 = CBAM(512)

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
        x = self.CBAM_layer1(x)

        x = self.layer4(x)

        x = self.CBAM_layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x