import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, swin_v2_t, Swin_V2_T_Weights

from model_builder.resnet_BAM import Resnet50_BAM
from model_builder.resnet_BCAM import Resnet50_BCBAM  
from model_builder.resnet_CBAM import Resnet50_CBAM
from model_builder.resnet_Swin import Resnet50_Swin, Resnet50_Swin_BAM, Resnet50_Swin_CBAM   

class Model(nn.Module):
    def __init__(self, num_classes, model_type, relu_replace=False):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.relu_replace = relu_replace
        self.model = self.build_model() 
    def replace_relu(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, name, nn.SiLU(inplace=True))
            else:
                self.replace_relu(child)
    def build_model(self):
        match self.model_type:
            case 0: #Swin
                swinv2Weight = Swin_V2_T_Weights.DEFAULT
                model = swin_v2_t(weights=swinv2Weight)

                in_features = model.head.in_features #768
                # model.head = nn.Linear(in_features, self.num_classes, bias=True)
                model.head = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, 768),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(768, self.num_classes)
                )
                print("Training on Swin architecture")
                return model
            case 1: #Default Resnet50
                resnet_weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=resnet_weights)
                activation = nn.ReLU()
                if self.relu_replace:
                    self.replace_relu(model)
                    activation = nn.SiLU()
                in_features = model.fc.in_features #2048
                fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    activation,
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )
                model.fc = fc
                print("Training on Resnet50 architecture")
                return model
            case 2: #Resnet50 Bam
                model = Resnet50_BAM(num_classes=self.num_classes)
                print("Training on Resnet50 with BAM")
                return model
            case 3: #Resnet50 CBAM
                model = Resnet50_CBAM(num_classes=self.num_classes)
                print("Training on Resnet50 with CBAM")
                return model
            case 4: #Resnet50 BCAM
                model = Resnet50_BCBAM(num_classes=self.num_classes)
                print("Training on Resnet50 with BCBAM")
                return model
            case 5: #Resnet50 Swin
                model = Resnet50_Swin(num_classes=self.num_classes)
                print("Training on Resnet50 and Swin")
                return model
            case 6: #Resnet50 Swin with BAM
                model = Resnet50_Swin_BAM(num_classes=self.num_classes)
                print("Training on Resnet50 and Swin with BAM")
                return model
            case 7: #Wider Resnet Swin with BAM
                model = Resnet50_Swin_CBAM(num_classes=self.num_classes)
                print("Training on Wider Resnet50 and Swin with CBAM")
                return model
    def register_hook(self, hook_fn):
        match self.model_type:
            case 1: 
                self.model.layer4.feature_maps = None
                hook_handle = self.model.layer4.register_forward_hook(hook_fn)
                return hook_handle
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
            case 5:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
                return hook_handle
            case 6:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
                return hook_handle
            case 7:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
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
            case 5:
                return self.model.fusion.feature_maps
            case 6:
                return self.model.fusion.feature_maps
            case 7:
                return self.model.fusion.feature_maps
    def forward(self, x):
        return self.model(x)
    