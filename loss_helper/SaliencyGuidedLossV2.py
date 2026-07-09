import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from timm.loss.cross_entropy import SoftTargetCrossEntropy

from monai.losses.tversky import TverskyLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SaliencyGuidedLossV2(nn.Module):
    def __init__(self, type, enabled_batchwise_transform=False, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super(SaliencyGuidedLossV2, self).__init__()

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
        ).to(device)
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
                return SoftTargetCrossEntropy()
        return nn.CrossEntropyLoss()

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
    def compute_sigmoid_focal_loss(self, attention_map, binary_masks):
        loss = self.sigmoid_focal_loss(
            attention_map,
            binary_masks)
        return loss
    def log_cosh(self, x):
        return torch.log(torch.cosh(x).clamp(min=1e-12))

    def compute_tversky_loss(self, attention_map, binary_masks):
        loss = self.tversky_loss(
            attention_map,
            binary_masks.float()
        )
        return loss
    
    def compute_tv_loss(self, attention_map):
        """
        attention_map: [B, 1, H, W] (logits)
        """
        # Convert logits → probabilities (stabilizes TV)
        attn = torch.sigmoid(attention_map)

        # Vertical differences
        diff_h = torch.abs(attn[:, :, 1:, :] - attn[:, :, :-1, :])

        # Horizontal differences
        diff_w = torch.abs(attn[:, :, :, 1:] - attn[:, :, :, :-1])

        # Normalize by total number of elements
        tv_loss = (diff_h.sum() + diff_w.sum()) / attn.numel()

        return tv_loss

    def forward(self, pred, target, feature_maps, binary_masks, has_masks, epoch):
        # 1. Standard Classification Loss
        cls_loss = self.classification_loss(pred, target)

        # #create attention map
        # attention_map = self.create_attention_map(feature_maps=feature_maps, binary_masks=binary_masks)

        device = pred.device
        valid_idx = has_masks.bool()
        if valid_idx.any():
            # compute attention only for samples with masks
            feature_maps_valid = feature_maps[valid_idx]               # [n_valid, C, Hf, Wf]
            attention_valid = self.attention_head(feature_maps_valid)  # [n_valid,1,Ha,Wa]

            # ensure masks have channel dim and downsample to attention map size with AREA (soft)
            masks = binary_masks.unsqueeze(1) if binary_masks.dim() == 3 else binary_masks
            masks_valid = masks[valid_idx].float()                     # [n_valid,1,Hm,Wm]
            masks_small = F.interpolate(masks_valid, size=attention_valid.shape[2:], mode='area')

            # move to same device
            masks_small = masks_small.to(attention_valid.device)
            sigmoid_fc_loss = self.sigmoid_focal_loss(attention_valid, masks_small)
            tversky_loss = self.tversky_loss(attention_valid, masks_small)
        else:
            sigmoid_fc_loss = torch.zeros((), device=device)
            tversky_loss = torch.zeros((), device=device)

        # tv_loss = self.compute_tv_loss(attention_map)

        total_loss = self.alpha * cls_loss + self.beta * sigmoid_fc_loss + self.gamma * tversky_loss

        return total_loss, cls_loss, sigmoid_fc_loss, tversky_loss
    