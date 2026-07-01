import torch.nn as nn
import torch.nn as nn
from model_builder.hierarchy_model.Resnet50_Swin_Hier import Resnet50_Swin_Hier
from model_builder.hierarchy_model.Xception_Swin_Hier import Xception_Swin_Hier

from model_builder.hierarchy_model.hierarchy_models import Resnet50_Hierarchy, VGG16_Hierarchy   
class Model(nn.Module):
    def __init__(self, num_classes, model_type, attention_layer_type, replace_relu=False):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.attention_layer_type = attention_layer_type
        self.replace_relu = replace_relu
        self.model = self.build_model()
    def build_model(self):
        match self.model_type:
            case 3: #Resnet50
                model = Resnet50_Hierarchy(self.num_classes, self.attention_layer_type, self.replace_relu)
                print("Training on Resnet50 architecture")
                return model
            case 2: #VGG16
                model = VGG16_Hierarchy(self.num_classes, self.attention_layer_type, self.replace_relu)
                print("Training on VGG16 architecture")
                return model
            case 1: #Resnet Swin
                model = Resnet50_Swin_Hier(self.num_classes, self.attention_layer_type, self.replace_relu)
                print(f"Training on Resnet50 Swin Hier architecture with {attention_mode(self.attention_layer_type)}")
                return model
            case 3: #Xception Swin
                model = Xception_Swin_Hier(self.num_classes, self.attention_layer_type, self.replace_relu)
                print("Training on Xception Swin Hier architecture")
                return model
    def register_hook(self, hook_fn):
        match self.model_type:
            case 3:
                self.model.model_input[7].feature_maps = None
                hook_handle = self.model.model_input[7].register_forward_hook(hook_fn)
                return hook_handle
            case 2:
                self.model.features.feature_maps = None
                hook_handle = self.model.features.register_forward_hook(hook_fn)
                return hook_handle
            case 1:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
                return hook_handle
            case 4:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
                return hook_handle
    def get_feature_maps(self):
        match self.model_type:
            case 3:
                return self.model.model_input[7].feature_maps
            case 2:
                return self.model.features.feature_maps
            case 1:
                return self.model.fusion.feature_maps
            case 4:
                return self.model.fusion.feature_maps
    def forward(self, x):
        return self.model(x)
    
def attention_mode(attention_layer_type):
    match attention_layer_type:
        case 1:
            return "CBAM"
        case 2:
            return "BidirectionalAttentionModule with Aug"
        case 3:
            return "CBAM"