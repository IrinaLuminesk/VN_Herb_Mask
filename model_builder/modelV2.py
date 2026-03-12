import torch.nn as nn

from custom_resnet.resnet_BAM import Resnet50_BAM
from custom_resnet.resnet_BCAM import Resnet50_BCBAM  
from custom_resnet.resnet_CBAM import Resnet50_CBAM       
from custom_resnet.resnet_BoT import Resnet50_BoT
class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = self.build_model() 
    def build_model(self):
        match self.model_type:
            case 1: #Resnet50 Bam
                model = Resnet50_BAM(num_classes=self.num_classes)
                print("Training on Resnet50 with BAM")
                return model
            case 2: #Resnet50 CBAM
                model = Resnet50_CBAM(num_classes=self.num_classes)
                print("Training on Resnet50 with CBAM")
                return model
            case 3: #Resnet50 BCAM
                model = Resnet50_BCBAM(num_classes=self.num_classes)
                print("Training on Resnet50 with BCBAM")
                return model
            case 4: #Resnet50 with Bottleneck Transformer
                model = Resnet50_BoT(self.num_classes)
                print("Training on Resnet50 with BoT")
                return model 
    def register_hook(self, hook_fn):
        match self.model_type:
            case 1: 
                self.model.BAM_layer2.feature_maps = None
                hook_handle = self.model.BAM_layer2.register_forward_hook(hook_fn)
                return hook_handle
            case 2:
                self.model.CBAM_layer2.feature_maps = None
                hook_handle = self.model.CBAM_layer2.register_forward_hook(hook_fn)
                return hook_handle
            case 3:
                self.model.BCBAM_layer2.feature_maps = None
                hook_handle = self.model.BCBAM_layer2.register_forward_hook(hook_fn)
                return hook_handle
            case 4:
                self.model.layer4.feature_maps = None
                hook_handle = self.model.layer4.register_forward_hook(hook_fn)
                return hook_handle
    def get_feature_maps(self):
        match self.model_type:
            case 1:
                return self.model.BAM_layer2.feature_maps
            case 2:
                return self.model.CBAM_layer2.feature_maps
            case 3:
                return self.model.BCBAM_layer2.feature_maps
            case 4:
                return self.model.layer4.feature_maps
    def forward(self, x):
        return self.model(x)
    