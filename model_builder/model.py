import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights,\
    densenet201, DenseNet201_Weights,\
    vgg16, VGG16_Weights, \
    alexnet, AlexNet_Weights,\
    mobilenet_v2, MobileNet_V2_Weights, \
    swin_v2_b, Swin_V2_B_Weights, \
    efficientnet_b4, EfficientNet_B4_Weights, \
    vit_b_16, ViT_B_16_Weights, \
    inception_v3, Inception_V3_Weights
import timm         

class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = self.build_model() 
    def build_model(self):
        match self.model_type:
            case 1: #Resnet50
                resnet_weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=resnet_weights)

                in_features = model.fc.in_features #2048
                fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )
                model.fc = fc
                print("Training on Resnet50 architecture")
                return model
            case 2: #VGG16
                vgg16_weights = VGG16_Weights.DEFAULT
                model = vgg16(weights=vgg16_weights)

                # vgg16_classifier = list(model.classifier.children())[:6]
                in_features = model.classifier[0].in_features #25088

                model.classifier = nn.Sequential(
                    # *vgg16_classifier,
                    # nn.Linear(in_features, 2048, bias=True),
                    # nn.BatchNorm1d(2048),
                    # nn.ReLU(),
                    # nn.Dropout(0.4),
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on VGG16 architecture")
                return model
            
            case 3: #Xception
                model = timm.create_model(
                    'xception65',
                    pretrained=True
                )

                in_features = model.get_classifier().in_features
                
                Xception_classifier = model.head
                Xception_classifier = list(Xception_classifier.children())[:2]

                model.head = nn.Sequential(
                    *Xception_classifier,
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on Xception65 architecture")
                return model
            
            case 4: #EfficientNetB4
                model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

                in_features = model.classifier[1].in_features #1792

                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on EfficientNetB4 architecture")
                return model
            
            case 5: #DenseNet201
                densenet_Weights = DenseNet201_Weights.DEFAULT
                model = densenet201(weights=densenet_Weights)

                in_features = model.classifier.in_features #1920
                fc = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes),
                )
                model.classifier = fc
                print("Training on DenseNet201 architecture")
                return model
            
            case 6: #MobileNet
                mobilenetv2_weights = MobileNet_V2_Weights.DEFAULT
                model = mobilenet_v2(weights=mobilenetv2_weights)

                in_features = model.classifier[1].in_features #1280

                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on MobileNetV2 architecture")
                return model

            case 7: #AlexNet
                alexnet_weight = AlexNet_Weights.DEFAULT
                model = alexnet(weights= alexnet_weight)

                in_features = model.classifier[1].in_features #9216

                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on AlexNet architecture")
                return model
            
            case 8: #ViT
                ViTWeight = ViT_B_16_Weights.DEFAULT
                model = vit_b_16(weights=ViTWeight)
                
                in_features = model.heads[0].in_features #768

                # model.heads = nn.Linear(in_features, self.num_classes)
                model.heads = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, 1024),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on ViT architecture")
                return model

            case 9: #Swin transform
                swinv2Weight = Swin_V2_B_Weights.DEFAULT
                model = swin_v2_b(weights=swinv2Weight)

                in_features = model.head.in_features #1024
                # model.head = nn.Linear(in_features, self.num_classes, bias=True)
                model.head = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, 1024),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on Swin architecture")
                return model
            case 10: #Inception-v3
                inception_v3_weight = Inception_V3_Weights.DEFAULT
                model = inception_v3(weights=inception_v3_weight)

                in_features = model.fc.in_features #2048
                model.fc = nn.Sequential(
                    nn.Linear(in_features, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, self.num_classes)
                )
                print("Training on Inception V3 architecture")
                return model
    def forward(self, x):
        return self.model(x)
    