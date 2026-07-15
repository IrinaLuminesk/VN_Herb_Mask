import torch

import torch.nn as nn
import torch.nn.functional as F

from timm.loss.cross_entropy import SoftTargetCrossEntropy

from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss

from torchvision.ops import sigmoid_focal_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

class SaliencyGuidedLossV3(nn.Module):
    def __init__(self, type, enabled_batchwise_transform=False, saliency_type=1, alpha=1.0, beta=1.0, gamma=1.0):
        super(SaliencyGuidedLossV3, self).__init__()

        self.type = type
        self.enabled_batchwise_transform = enabled_batchwise_transform
        self.saliency_type=saliency_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classification_loss = self.classification_loss_builder()
        self.saliency_loss1, self.saliency_loss2 = self.saliency_loss_builder()
    def classification_loss_builder(self):
        if self.type == "train" and self.enabled_batchwise_transform == True:
                return SoftTargetCrossEntropy()
        return nn.CrossEntropyLoss()
    
    def saliency_loss_builder(self):
        if self.saliency_type == 1: #BCEWithLogitsLoss + Dice loss
            bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
            dice_loss = DiceLoss(
                mode="binary",
                from_logits=True,
                log_loss=False,
                smooth=0.0,
                eps=1e-7,
            )
            return bce_loss, dice_loss
        #Sigmoid Focal Loss + Tverky loss
        sigmoidfc_loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        tversky_loss = TverskyLoss(
            mode="binary",
            alpha=0.3,
            beta=0.7,
            smooth=1e-5,
            eps=1e-5,
            from_logits=True
        )
        return sigmoidfc_loss, tversky_loss
    
    def create_attention_map(self, feature_maps, binary_masks):
        # Attention map
        attention_map = feature_maps.square().mean(dim=1, keepdim=True)

        attention_map = feature_maps.norm(p=2, dim=1, keepdim=True)
        attention_map = F.interpolate(
            attention_map, 
            size=binary_masks.shape[2:], 
            mode='bilinear',
            align_corners=False
        )

        #Normalize logits
        attention_map = (
             attention_map - attention_map.mean(dim=(2, 3), keepdim=True)
             ) / (
             attention_map.std(dim=(2, 3), keepdim=True) + 1e-6
            )
        return attention_map
    

    def forward(self, pred, target, feature_maps, binary_masks, has_masks):
        # 1. Standard Classification Loss
        cls_loss = self.classification_loss(pred, target)

        valid_idx = has_masks.bool()
        if valid_idx.any():
            ids = valid_idx.nonzero(as_tuple=True)[0]           # 1D indices
            # ensure tensors on device (avoid copies inside hot loop)
            fm_valid = feature_maps.index_select(0, ids)
            masks_valid = binary_masks.index_select(0, ids)

            attention_map = self.create_attention_map(feature_maps=fm_valid, binary_masks=masks_valid)
        
            #2. Dùng để khuyến khích mô hình học các feature nằm trong mask
            saliency_loss1 = self.saliency_loss1(attention_map, masks_valid)

            #3. Dùng Dice để khuyến khích mô hình học các feature tổng quan thay vì chỉ tập chung vào một chỗ
            saliency_loss2 = self.saliency_loss2(
                attention_map,
                masks_valid,
            )
        else:
            saliency_loss1 = torch.zeros((), device=device)
            saliency_loss2 = torch.zeros((), device=device)



        total_loss = self.alpha * cls_loss + self.beta * saliency_loss1 + self.gamma * saliency_loss2

        return total_loss, cls_loss, saliency_loss1, saliency_loss2
    


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        return sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )