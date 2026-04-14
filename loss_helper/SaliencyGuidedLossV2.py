import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from timm.loss.cross_entropy import SoftTargetCrossEntropy

from monai.losses.tversky import TverskyLoss

device='cuda' if torch.cuda.is_available() else 'cpu'

class SaliencyGuidedLoss(nn.Module):
    def __init__(self, type, enabled_batchwise_transform=False, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super(SaliencyGuidedLoss, self).__init__()

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
            reduction="none"
        )
        self.tversky_loss = TverskyLoss(
            include_background=True,
            sigmoid=True,      # applies sigmoid internally
            alpha=0.3,
            beta=0.7,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction="none"
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
    def compute_sigmoid_focal_loss(self, attention_map, binary_masks, has_masks):
        if has_masks.any():
            
            sfcl = self.sigmoid_focal_loss(attention_map, binary_masks)  # [B,1,H,W]

            # mask samples without GT masks
            has_mask = has_masks.view(-1, 1).float()
            sigmoid_fc_loss = (sfcl * has_mask).sum() / has_mask.sum()
        else:
            sigmoid_fc_loss = torch.zeros((), device=device) #Yêu cầu mô hình cư xử bình thường
        return sigmoid_fc_loss
    def log_cosh(self, x):
        return torch.log(torch.cosh(x + 1e-12))

    def compute_tversky_loss(self, attention_map, binary_masks, has_masks):
        if has_masks.any():
            tversky_loss = self.tversky_loss(attention_map, binary_masks)

            has_mask = has_masks.view(-1, 1).float()
            tversky_loss = (tversky_loss * has_mask).sum() / has_mask.sum()
            tversky_loss_fn = self.log_cosh(tversky_loss)
        else:
            tversky_loss_fn = torch.zeros((), device=device)
        return tversky_loss_fn
    
    def compute_tv_loss(self, attention_map, has_masks):
        """
        attention_map: [B, 1, H, W] (LOGITS)
        has_masks:     [B] boolean tensor
        """
        if has_masks.any():
            valid_idx = has_masks.nonzero(as_tuple=True)[0]
            attn = attention_map[valid_idx]  # only supervised samples

            # vertical differences
            diff_h = torch.abs(attn[:, :, 1:, :] - attn[:, :, :-1, :])

            # horizontal differences
            diff_w = torch.abs(attn[:, :, :, 1:] - attn[:, :, :, :-1])

            tv_loss = diff_h.mean() + diff_w.mean()

        else:
            tv_loss = torch.zeros(
                (),
                device=attention_map.device
            )

        return tv_loss


    def forward(self, pred, target, feature_maps, binary_masks, has_masks, epoch):
        # 1. Standard Classification Loss
        cls_loss = self.classification_loss(pred, target)

        #create attention map
        attention_map = self.create_attention_map(feature_maps=feature_maps, binary_masks=binary_masks)
        
        #2. Dùng để khuyến khích mô hình học các feature nằm trong mask
        sigmoid_fc_loss = self.compute_sigmoid_focal_loss(attention_map, binary_masks, has_masks)

        #3. Dùng Dice để khuyến khích mô hình học các feature tổng quan thay vì chỉ tập chung vào một chỗ
        tversky_loss_fn = self.compute_tversky_loss(attention_map, binary_masks, has_masks)

        #4 Total variation loss
        tv_loss = self.compute_tv_loss(attention_map, has_masks)

        total_loss = self.alpha * cls_loss + self.beta * sigmoid_fc_loss + self.gamma * tversky_loss_fn + self.delta * tv_loss 

        return total_loss, cls_loss, sigmoid_fc_loss, tversky_loss_fn, tv_loss
    