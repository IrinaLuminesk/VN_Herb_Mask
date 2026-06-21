import torch
import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
import timm
from attention_module.attention import Attention_Layer


from attention_module.helper_layers import CNNtoSwinAdapter, FusionBlock
from utils.Utilities import replace_relu_helper

class Xception_Swin_Hier(nn.Module):
    def __init__(self, num_classes, attention_layer_type, replace_relu=False):
        super().__init__()
        self.num_classes = num_classes
        self.attention_layer_type = attention_layer_type
        self.replace_relu = replace_relu
        self.build_layers()
    def build_layers(self):
            backbone_model = timm.create_model("xception65.tf_in1k", pretrained=True)

            activation = nn.ReLU()
            if self.replace_relu:
                replace_relu_helper(backbone_model)
                activation = nn.SiLU()
                print("Replacing RELU with SILU")

            swin_weights = Swin_V2_B_Weights.DEFAULT
            swin_model = swin_v2_b(weights=swin_weights)

            self.model_input = nn.Sequential(*backbone_model.stem)
            #Trong Xception 65 có 21 XceptionModule sau stem
            self.block0to19 = nn.Sequential(*backbone_model.blocks[:20])

            #Sau block20 có một layer attention trước khi chia ra 2 nhánh
            self.attention_layer1 = Attention_Layer(self.attention_layer_type, channels=1024, replace_relu=self.replace_relu)
            # self.BAM_layer1 = BidirectionalAttentionModule(1024, replace_relu=self.relu_replace)


            self.block20 = backbone_model.blocks[20]
            self.attention_layer2_branchA = Attention_Layer(self.attention_layer_type, channels=2048, replace_relu=self.replace_relu)
            # self.BAM_layer2_branchA = BidirectionalAttentionModule(2048, replace_relu=self.relu_replace)

            #Swin
            self.swin_layer = swin_model.features[7]
            self.adapt_cnn_2_Swin = CNNtoSwinAdapter(pool_layer=False) #Không cần đổi do output của block0to19 là [1024, 8, 8], chỉ cần đổi thức tự
            # self.attention_layer2_branchB = Attention_Layer(self.attention_layer_type, channels=1024, replace_relu=self.replace_relu)
            # self.BAM_layer2_branchB = BidirectionalAttentionModule(channels=1024, replace_relu=self.relu_replace)


            #Fusion
            self.fusion = FusionBlock()
            self.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),   # (N, C, H, W) -> (N, C, 1, 1)
                nn.Flatten(1)              # (N, C, 1, 1) -> (N, C)
            )
            self.fc = nn.ModuleDict()
            for key, value in self.num_classes.items():
                self.fc[key] = nn.Sequential(
                     nn.Linear(3072, 1024),
                     nn.BatchNorm1d(1024),
                     activation,
                     nn.Dropout(0.4),
                     nn.Linear(1024, value),
                )

    def augment_feature(self, x):
        if self.training: #Biến này kế thừa
            noise = 0.01 * torch.randn_like(x)
            return x + noise

        return x
    def forward(self, x):
        x = self.model_input(x)
        #Layer này chia nhánh ra 2 nhánh, nhánh 1 vào layer20 gốc của Xception65 và nhánh 2 vào stage 3 và 4 của Swin
        shared = self.block0to19(x) #(1024, 8, 8)
        if self.attention_layer_type != 1:
            shared_aug = self.augment_feature(shared)
            shared = self.attention_layer1(shared, shared_aug)
        else:
            shared = self.attention_layer1(shared)

        #Branch A
        xception_branch = self.block20(shared) #(2048, 8, 8)
        if self.attention_layer_type != 1:
            xception_branch_aug = self.augment_feature(xception_branch)
            xception_branch = self.attention_layer2_branchA(xception_branch, xception_branch_aug) #output (2048, 8, 8)
        else:
            xception_branch = self.attention_layer2_branchA(xception_branch)


        #Branch 2
        swin_branch = self.adapt_cnn_2_Swin(shared)  # BCHW -> BHWC
        swin_branch = self.swin_layer(swin_branch) #Output [B, 8, 8, 1024]
        swin_branch = swin_branch.permute(0, 3, 1, 2).contiguous() #Output [B, 1024, 8, 8]
        # if self.attention_layer_type != 1:
        #     swin_branch_aug = self.augment_feature(swin_branch)
        #     swin_branch = self.attention_layer2_branchB(swin_branch, swin_branch_aug) #Output [B, 1024, 8, 8]
        # else:
        #     swin_branch = self.attention_layer2_branchB(swin_branch)

        Fused = self.fusion(xception_branch, swin_branch)
        x = self.avgpool(Fused)
        x = torch.flatten(x, 1)
        logits_dict = dict()
        for key, _ in self.num_classes.items():
           logits_dict[key] = self.fc[key](x) 
        return logits_dict