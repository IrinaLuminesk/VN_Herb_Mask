import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.cross_entropy import SoftTargetCrossEntropy

from segmentation_models_pytorch.losses import DiceLoss

class SaliencyGuidedLoss(nn.Module):
    def __init__(self, type, enabled_batchwise_transform=False, alpha=1.0, beta=1.0, gamma=1.0):
        super(SaliencyGuidedLoss, self).__init__()

        self.type = type
        self.enabled_batchwise_transform = enabled_batchwise_transform
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classification_loss = self.loss_builder()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_loss = DiceLoss(
            mode="binary",
            from_logits=True,
            log_loss=False,
            smooth=0.0,
            eps=1e-7,
        )
    def loss_builder(self):
        if self.type == "train" and self.enabled_batchwise_transform == True:
                return SoftTargetCrossEntropy()
        return nn.CrossEntropyLoss()

    def create_attention_map(self, feature_maps, binary_masks):
        # Attention map
        attention_map = torch.mean(feature_maps, dim=1, keepdim=True)
        attention_map = F.interpolate(
            attention_map, 
            size=binary_masks.shape[2:], 
            mode='bilinear',
            align_corners=False
        )

        #Normalize logits
        attention_map = attention_map / (
            attention_map.std(dim=(2, 3), keepdim=True) + 1e-6
        )
        return attention_map
    def BCE_loss(self, attention_map, binary_masks, has_masks):

        # ---- 4. Alignment loss (only if masks exist) ----
        if has_masks.any():
            # BCE per pixel
            bce = self.bce_loss(attention_map, binary_masks)  # [B,1,H,W]

            # mean over spatial dimensions
            bce = bce.mean(dim=(2, 3))  # [B,1]

            # mask samples without GT masks
            has_mask = has_masks.view(-1, 1).float()
            bce_align_loss = (bce * has_mask).sum() / (has_mask.sum() + 1e-8)
        else:
            bce_align_loss = torch.zeros((), device='cuda' if torch.cuda.is_available() else 'cpu')
        return bce_align_loss
    
    def compute_dice_loss(self, attention_map, binary_masks, has_masks):
        if has_masks.any():
            valid_idx = has_masks.nonzero(as_tuple=True)[0]

            dice_loss = self.dice_loss(
                attention_map[valid_idx],
                binary_masks[valid_idx],
            )
        else:
            dice_loss = torch.zeros((), device='cuda' if torch.cuda.is_available() else 'cpu')
        return dice_loss

    def forward(self, pred, target, feature_maps, binary_masks, has_masks):
        # 1. Standard Classification Loss
        cls_loss = self.classification_loss(pred, target)

        #create attention map
        attention_map = self.create_attention_map(feature_maps=feature_maps, binary_masks=binary_masks)
        
        #2. Dùng để khuyến khích mô hình học các feature nằm trong mask
        bce_align_loss = self.BCE_loss(attention_map, binary_masks, has_masks)

        #3. Dùng Dice để khuyến khích mô hình học các feature tổng quan thay vì chỉ tập chung vào một chỗ
        dice_loss = self.compute_dice_loss(attention_map, binary_masks, has_masks)

        total_loss = self.alpha * cls_loss + self.beta * bce_align_loss + self.gamma * dice_loss

        return total_loss
    