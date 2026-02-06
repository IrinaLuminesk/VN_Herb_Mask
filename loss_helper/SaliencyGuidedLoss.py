import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.cross_entropy import SoftTargetCrossEntropy

class SaliencyGuidedLoss(nn.Module):
    def __init__(self, type, enabled_batchwise_transform=False,alpha=1.0):
        super(SaliencyGuidedLoss, self).__init__()
        # self.alpha = alpha  # Weight of the attention guidance
        # self.ce_loss = nn.CrossEntropyLoss()

        self.type = type
        self.enabled_batchwise_transform = enabled_batchwise_transform
        self.alpha = alpha
        self.first_loss = self.loss_builder()
        self.bce_loss = nn.BCELoss(reduction='none')
    def loss_builder(self):
        if self.type == "train" and self.enabled_batchwise_transform == True:
                return SoftTargetCrossEntropy()
        return nn.CrossEntropyLoss()

    def forward(self, pred, target, feature_maps, binary_masks, has_masks):
        # 1. Standard Classification Loss
        cls_loss = self.first_loss(pred, target)

        # Attention map
        attention_map = torch.mean(feature_maps, dim=1, keepdim=True)
        attention_map = F.interpolate(
            attention_map, 
            size=binary_masks.shape[2:], 
            mode='bilinear',
            align_corners=False
        )

        attention = torch.sigmoid(attention_map)

        if has_masks.any():
            has_mask = has_masks.view(-1, 1, 1, 1).float()
            bce_loss = self.bce_loss(attention_map, binary_masks)
            align_loss = (bce_loss * has_mask).sum() / (has_mask.sum() + 1e-8)
        else:
            align_loss = torch.tensor(0.0, device=pred.device)

        return cls_loss + self.alpha * align_loss