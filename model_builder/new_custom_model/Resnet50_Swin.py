import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, swin_v2_b, Swin_V2_B_Weights
from attention_module.attention import Attention_Layer


from attention_module.helper_layers import CNNtoSwinAdapter, FusionBlock
from utils.Utilities import replace_relu_helper

class Resnet50_Swin(nn.Module):
    def __init__(self, num_classes, attention_layer_type, replace_relu=False):
        super().__init__()
        self.num_classes = num_classes
        self.attention_layer_type = attention_layer_type
        self.replace_relu = replace_relu
        self.build_layers()
    def build_layers(self):
            resnet_weights = ResNet50_Weights.DEFAULT
            backbone_model = resnet50(weights=resnet_weights)

            activation = nn.ReLU()
            if self.replace_relu:
                replace_relu_helper(backbone_model)
                activation = nn.SiLU()
                print("Replacing RELU with SILU")

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

            #Sau layer 3 có một layer BAM trước khi chia ra 2 nhánh
            self.attention_layer1 = Attention_Layer(self.attention_layer_type, channels=1024, replace_relu=self.replace_relu)
            # self.BAM_layer1 = BidirectionalAttentionModule(1024, replace_relu=self.relu_replace)


            self.layer4 = backbone_model.layer4
            self.attention_layer2_branchA = Attention_Layer(self.attention_layer_type, channels=2048, replace_relu=self.replace_relu)
            # self.BAM_layer2_branchA = BidirectionalAttentionModule(2048, replace_relu=self.relu_replace)

            #Swin
            self.swin_layer = swin_model.features[7]
            self.adapt_cnn_2_Swin = CNNtoSwinAdapter(pool_layer=True, kernel_size=2) #Dùng để đổi [B, 16, 16, 1024] sang [B, 1024, 8, 8]
            self.attention_layer2_branchB = Attention_Layer(self.attention_layer_type, channels=1024, replace_relu=self.replace_relu)
            # self.BAM_layer2_branchB = BidirectionalAttentionModule(channels=1024, replace_relu=self.relu_replace)


            #Fusion
            self.fusion = FusionBlock()
            self.avgpool = backbone_model.avgpool
            self.fc = nn.Sequential(
                nn.Linear(3072, 1024),
                nn.BatchNorm1d(1024),
                activation,
                nn.Dropout(0.4),
                nn.Linear(1024, self.num_classes),
            )

    def augment_feature(self, x):
        if self.training: #Biến này kế thừa
            noise = 0.01 * torch.randn_like(x)
            return x + noise

        return x
    def forward(self, x):
        x = self.model_input(x)
        x = self.layer1(x)
        x= self.layer2(x)
        #Layer này chia nhánh ra 2 nhánh, nhánh 1 vào layer4 gốc của Resnet và nhánh 2 vào stage 3 và 4 của Swin
        shared = self.layer3(x) #(1024, 16, 16)
        if self.attention_layer_type != 1:
            shared_aug = self.augment_feature(shared)
            shared = self.attention_layer1(shared, shared_aug)
        else:
            shared = self.attention_layer1(shared)

        #Branch A
        resnet_branch = self.layer4(shared) #(2048, 8, 8)
        if self.attention_layer_type != 1:
            resnet_branch_aug = self.augment_feature(resnet_branch)
            resnet_branch = self.attention_layer2_branchA(resnet_branch, resnet_branch_aug) #output (2048, 8, 8)
        else:
            resnet_branch = self.attention_layer2_branchA(resnet_branch)


        #Branch 2
        swin_branch = self.adapt_cnn_2_Swin(shared)  # BCHW -> BHWC
        swin_branch = self.swin_layer(swin_branch) #Output [B, 8, 8, 1024]
        swin_branch = swin_branch.permute(0, 3, 1, 2).contiguous() #Output [B, 1024, 8, 8]
        if self.attention_layer_type != 1:
            swin_branch_aug = self.augment_feature(swin_branch)
            swin_branch = self.attention_layer2_branchB(swin_branch, swin_branch_aug) #Output [B, 1024, 8, 8]
        else:
            swin_branch = self.attention_layer2_branchB(swin_branch)

        Fused = self.fusion(resnet_branch, swin_branch)
        x = self.avgpool(Fused)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x