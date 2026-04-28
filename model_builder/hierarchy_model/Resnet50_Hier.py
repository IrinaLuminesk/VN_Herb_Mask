import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

import copy

from model_builder.hierarchy_model.DOHF import DOHF

class Resnet50_Hier(nn.Module):
    def __init__(self, hierarchy):
        # hierarchy = {species: 200, genus: }
        self.hierarchy = hierarchy
        self.build_layers()
    def build_layers(self):
        resnet_weight = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=resnet_weight)
        self.backbone_model = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        self.branch1 = copy.deepcopy(backbone.layer4) 
        self.branch2 = copy.deepcopy(backbone.layer4)
        self.branch3 = copy.deepcopy(backbone.layer4)
        self.branch4 = copy.deepcopy(backbone.layer4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.ModuleList([
            nn.Linear(2048, 1024),
            nn.Linear(2048, 1024),
            nn.Linear(2048, 1024),
            nn.Linear(2048, 1024),
        ])
        
        self.classifiers = nn.ModuleDict({
            "order": nn.Linear(1024, self.hierarchy["order"]),
            "family": nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.hierarchy["family"])
            ),
            "genus": nn.Sequential(
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, self.hierarchy["genus"])
            ),
            "species": nn.Sequential(
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, self.hierarchy["species"])
            ),})
        
        self.dohf = DOHF()

    def forward(self, x):
        x = self.backbone_model(x)

        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)

        feats = [f1, f2, f3, f4]

        vectors = []

        for i, f in enumerate(feats):
            v = self.pool(f).flatten(1)   # (B, 2048)
            v = self.proj[i](v)           # (B, 1024)
            v = F.normalize(v, dim=-1)
            vectors.append(v)

        refined = []
        shallow = None

        for v in vectors:
            v = self.dohf(shallow, v)
            refined.append(v)
            shallow = v

        outputs = {}
        for v, h in zip(refined, reversed(self.hierarchy)):
            outputs[h] = self.classifiers[h](v)

        return outputs