import torch.nn as nn

from model_builder.new_custom_model.Resnet50_Swin import Resnet50_Swin
from model_builder.new_custom_model.Xception_Swin import Xception_Swin

class Model(nn.Module):
    def __init__(self, num_classes, model_type, attention_layer_type, replace_relu=False):
        self.num_classes = num_classes
        self.model_type = model_type
        self.attention_layer_type = attention_layer_type
        self.replace_relu = replace_relu
        self.model = self.build_model()
    def build_model(self):
        match self.model_type:
            case 1: #Resnet50 Swin
                model = Resnet50_Swin(
                    num_classes=self.num_classes, 
                    attention_layer_type=self.attention_layer_type,
                    replace_relu=self.replace_relu
                )
                print(f"Training on Resnet50 and Swin with {attention_mode(self.attention_layer_type)}")
                return model
            case 2: #Xception65 Swin
                model = Xception_Swin(
                    num_classes=self.num_classes,
                    attention_layer_type=self.attention_layer_type,
                    replace_relu=self.replace_relu
                )
                print(f"Training on Xception 65 and Swin with {attention_mode(self.attention_layer_type)}")
                return model
    def register_hook(self, hook_fn):
        match self.model_type:
            case 1:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
                return hook_handle
            case 2:
                self.model.fusion.feature_maps = None
                hook_handle = self.model.fusion.register_forward_hook(hook_fn)
                return hook_handle
    def get_feature_maps(self):
        match self.model_type:
            case 1:
                return self.model.fusion.feature_maps
            case 2:
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