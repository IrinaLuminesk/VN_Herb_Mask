import torch.nn as nn
import torch.nn as nn

from model_builder.hierarchy_model.hierarchy_models import Resnet50_Hierarchy, VGG16_Hierarchy   
class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = self.build_model()
    def build_model(self):
        match self.model_type:
            case 1: #Resnet50
                model = Resnet50_Hierarchy(self.num_classes, 0, False)
                print("Training on Resnet50 architecture")
                return model
            case 2: #VGG16
                model = VGG16_Hierarchy(self.num_classes, 0, False)
                print("Training on VGG16 architecture")
                return model
    def forward(self, x):
        return self.model(x)