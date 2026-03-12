import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from custom_resnet.resnet_BAM import Resnet18_BAM
from custom_resnet.resnet_BCAM import Resnet18_BCBAM  
from custom_resnet.resnet_CBAM import Resnet18_CBAM       

class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = self.build_model() 
    def build_model(self):
        match self.model_type:
            case 1: #Resnet18 Normal
                resnet_weights = ResNet18_Weights.DEFAULT
                model = resnet18(weights=resnet_weights)
                in_features = model.fc.in_features #512
                fc = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, self.num_classes),
                )
                model.fc = fc
                print("Training on Resnet18 architecture")
                return model
            case 2: #Resnet50 Bam
                model = Resnet18_BAM(num_classes=self.num_classes)
                print("Training on Resnet18 with BAM")
                return model
            case 3: #Resnet50 CBAM
                model = Resnet18_CBAM(num_classes=self.num_classes)
                print("Training on Resnet18 with CBAM")
                return model
            case 4: #Resnet50 BCAM
                model = Resnet18_BCBAM(num_classes=self.num_classes)
                print("Training on Resnet18 with BCBAM")
                return model
    def register_hook(self, hook_fn):
        match self.model_type:
            case 1: 
                self.model.layer4.feature_maps = None
                hook_handle = self.model.layer4.register_forward_hook(hook_fn)
            case 2: 
                self.model.BAM_layer2.feature_maps = None
                hook_handle = self.model.BAM_layer2.register_forward_hook(hook_fn)
                return hook_handle
            case 3:
                self.model.CBAM_layer2.feature_maps = None
                hook_handle = self.model.CBAM_layer2.register_forward_hook(hook_fn)
                return hook_handle
            case 4:
                self.model.BCBAM_layer2.feature_maps = None
                hook_handle = self.model.BCBAM_layer2.register_forward_hook(hook_fn)
                return hook_handle
    def get_feature_maps(self):
        match self.model_type:
            case 1:
                return self.model.layer4.feature_maps
            case 2:
                return self.model.BAM_layer2.feature_maps
            case 3:
                return self.model.CBAM_layer2.feature_maps
            case 4:
                return self.model.BCBAM_layer2.feature_maps
    def forward(self, x):
        return self.model(x)
    