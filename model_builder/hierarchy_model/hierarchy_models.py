import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
from attention_module.attention import Attention_Layer

from utils.Utilities import replace_relu_helper

class Resnet50_Hierarchy(nn.Module):
    def __init__(self, num_classes, attention_layer_type, replace_relu=False):
        super().__init__()
        self.num_classes = num_classes #This is a dict
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
            
            self.model_input = nn.Sequential(
                backbone_model.conv1,
                backbone_model.bn1,
                backbone_model.relu,
                backbone_model.maxpool,
                backbone_model.layer1,
                backbone_model.layer2,
                backbone_model.layer3,
                backbone_model.layer4
            )

            #Layer này sau layer 4 của resnet50
            self.attention_layer = Attention_Layer(self.attention_layer_type, channels=2048, replace_relu=self.replace_relu)

            self.avgpool = backbone_model.avgpool
            self.fc = nn.ModuleDict()
            for key, value in self.num_classes.items():
                self.fc[key] = nn.Sequential(
                     nn.Linear(2048, 1024),
                     nn.BatchNorm1d(1024),
                     activation,
                     nn.Dropout(0.4),
                     nn.Linear(1024, value),
                )
    def forward(self, x):
        x = self.model_input(x)
        x = self.attention_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits_dict = dict()
        for key, _ in self.num_classes.items():
           logits_dict[key] = self.fc[key](x) 
        return logits_dict