from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.cross_entropy import SoftTargetCrossEntropy

from segmentation_models_pytorch.losses import TverskyLoss

from torchvision.ops import sigmoid_focal_loss

class HierarchySaliencyGuidedLossV2(nn.Module):
    def __init__(self, 
                 num_classes,
                 type, 
                 enabled_batchwise_transform=False, 
                 #Dùng cho Saliency
                 alpha=1.0, 
                 beta=1.0, 
                 gamma=1.0, 
                 delta=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.type = type
        self.enabled_batchwise_transform = enabled_batchwise_transform
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.classification_loss = self.classification_loss_builder()
        self.saliency_loss1, self.saliency_loss2 = self.saliency_loss_builder()
        self.consistency_loss = nn.KLDivLoss(
            reduction="batchmean"
        )

    def classification_loss_builder(self):
        if self.type == "train" and self.enabled_batchwise_transform == True:
            return {
                key: SoftTargetCrossEntropy()
                for key in self.num_classes
            }
        return {
            key: nn.CrossEntropyLoss()
            for key in self.num_classes
        }

    def saliency_loss_builder(self):
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
    
    def compute_classification_loss(self, logits, targets, device):
        total_class_loss = torch.tensor(0.0, device=device)
        loss = dict()
        for idx, key in enumerate(self.num_classes):
            logit = logits[key]
            if self.type == "train" and self.enabled_batchwise_transform == True:
                target = targets[key]
            else:
                target = targets[:, idx]
            loss[key] = self.classification_loss[key](logit, target)
            total_class_loss += loss[key]
        return loss, total_class_loss
    
    def compute_consistent_loss(self, logits, hier_matrixs, device):
        consistency_loss = torch.tensor(0.0, device=device)
        loss = dict()
        for key in hier_matrixs.keys():
            x_name, y_name = key.split("2")
            x = torch.softmax(logits[x_name], dim=1)
            y = torch.softmax(logits[y_name], dim=1)
            loss_xy = loss[key] = self.consistency_loss(x.log(), y @ hier_matrixs[key])
            consistency_loss += loss_xy
        return loss, consistency_loss

    def forward(self, logits, targets, feature_maps, binary_masks, has_masks, hier_matrixs):
        #Logits là một dict chứa các phân cấp, 
        #target là một batch chứa các array của từng phân cấp
        #hier_matrixs là một dict chứa thông tin liên hệ từng phân cấp
        #family -> genus
        #genus -> species
        if self.type == "train" and self.enabled_batchwise_transform == True:
            device = next(iter(targets.values())).device
        else:
            device = targets.device

        # 1. Standard Classification Loss
        #each_classification_loss là một dict chứa các loss của từng cấp
        #classification_loss là loss đã cộng hết các cấp 
        each_classification_loss, classification_loss = self.compute_classification_loss(logits, targets, device)

        #each_consistent_loss là một dict chứa các loss của từng cấp
        #consistent_loss là loss đã cộng hết các cấp 
        each_consistent_loss, consistent_loss = self.compute_consistent_loss(logits, hier_matrixs, device)

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

        total_loss = self.alpha * classification_loss + self.beta * saliency_loss1 + self.gamma * saliency_loss2 + self.delta * consistent_loss 

        return (classification_loss, 
                each_classification_loss, 
                consistent_loss, 
                each_consistent_loss, 
                saliency_loss1, 
                saliency_loss2,
                total_loss)

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