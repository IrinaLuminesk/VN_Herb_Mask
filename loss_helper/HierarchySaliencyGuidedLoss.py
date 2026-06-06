import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from timm.loss.cross_entropy import SoftTargetCrossEntropy

from monai.losses.tversky import TverskyLoss

class HierarchySaliencyGuidedLoss(nn.Module):
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
        # in_channels sẽ được hàm tự bỏ vào sau
        self.attention_head = nn.LazyConv2d(
            out_channels=1,
            kernel_size=1,
            bias=False
        ).to("cuda")
        self.classification_loss = self.loss_builder()
        self.sigmoid_focal_loss = lambda logits, targets: sigmoid_focal_loss(
            logits, 
            targets,  
            alpha=0.25,
            gamma=2.0,
            reduction="mean"
        )
        self.tversky_loss = TverskyLoss(
            to_onehot_y=False,
            include_background=True,
            sigmoid=True,      # applies sigmoid internally
            alpha=0.3,
            beta=0.7,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction="mean"
        )
    def loss_builder(self):
        if self.type == "train" and self.enabled_batchwise_transform == True:
            return {
                key: SoftTargetCrossEntropy()
                for key in self.num_classes
            }
        return {
            key: nn.CrossEntropyLoss()
            for key in self.num_classes
        }
    
    def create_attention_map(self, feature_maps, binary_masks):
        # Attention map
        # attention_map = torch.mean(feature_maps, dim=1, keepdim=True)

        attention_map = self.attention_head(feature_maps)
        attention_map = F.interpolate(
            attention_map, 
            size=binary_masks.shape[2:], 
            mode='bilinear',
            align_corners=False
        )
        return attention_map
    def compute_sigmoid_focal_loss(self, attention_map, binary_masks, valid_idx):
        loss = self.sigmoid_focal_loss(
            attention_map[valid_idx],
            binary_masks[valid_idx])
        return loss
    
    def compute_tversky_loss(self, attention_map, binary_masks, valid_idx):
        loss = self.tversky_loss(
            attention_map[valid_idx],
            binary_masks[valid_idx].float()
        )
        return loss
    
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