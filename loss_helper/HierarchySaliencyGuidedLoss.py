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
        self.consistency_loss = nn.KLDivLoss(
            reduction="batchmean"
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

        #create attention map
        attention_map = self.create_attention_map(feature_maps=feature_maps, binary_masks=binary_masks)

        valid_idx = has_masks.bool()
        if valid_idx.any():

        
            sigmoid_fc_loss = self.compute_sigmoid_focal_loss(attention_map, binary_masks, valid_idx)

            
            tversky_loss = self.compute_tversky_loss(attention_map, binary_masks, valid_idx)

        else:
            sigmoid_fc_loss = torch.zeros((), device=device)
            tversky_loss = torch.zeros((), device=device)


        total_loss = self.alpha * classification_loss + self.beta * sigmoid_fc_loss + self.gamma * tversky_loss + self.delta * consistent_loss 

        return (classification_loss, 
                each_classification_loss, 
                consistent_loss, 
                each_consistent_loss, 
                sigmoid_fc_loss, 
                tversky_loss,
                total_loss)