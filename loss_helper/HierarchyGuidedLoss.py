import torch
import torch.nn as nn
from timm.loss.cross_entropy import SoftTargetCrossEntropy

class HierarchyGuidedLoss(nn.Module):
    def __init__(self, num_classes, type, enabled_batchwise_transform=False):
        super(HierarchyGuidedLoss, self).__init__()
        self.num_classes = num_classes
        self.type = type
        self.enabled_batchwise_transform = enabled_batchwise_transform
        self.classification_loss = self.loss_builder()
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
    def compute_classification_loss(self, logits, targets):
        total_class_loss = torch.tensor(0.0, device=targets.device)
        loss = dict()
        for idx, key in enumerate(self.num_classes):
            logit = logits[key]
            target = targets[:, idx]
            loss[key] = self.classification_loss[key](logit, target)
            total_class_loss += loss[key]
        return loss, total_class_loss
    
    def compute_consistent_loss(self, logits, hier_matrixs, device):
        consistency_loss = torch.tensor(0.0, device=device)
        loss = dict()
        for key in hier_matrixs.keys():
            x_name, y_name = key.split("2")
            x = torch.softmax(logits[x_name], dim=0)
            y = torch.softmax(logits[y_name], dim=0)
            xy_score = torch.einsum("bi,ij,bj->b", x, hier_matrixs[key], y)
            loss_xy = loss[key] = -torch.log(xy_score + 1e-8).mean()
            consistency_loss += loss_xy
        return loss, consistency_loss
    
    def forward(self, logits, targets, hier_matrixs): 
        #Logits là một dict chứa các phân cấp, 
        #target là một batch chứa các array của từng phân cấp
        #hier_matrixs là một dict chứa thông tin liên hệ từng phân cấp
        #family -> genus
        #genus -> species
        
        #each_classification_loss là một dict chứa các loss của từng cấp
        #classification_loss là loss đã cộng hết các cấp 
        each_classification_loss, classification_loss = self.compute_classification_loss(logits, targets)

        #each_consistent_loss là một dict chứa các loss của từng cấp
        #consistent_loss là loss đã cộng hết các cấp 
        device = targets.device
        each_consistent_loss, consistent_loss = self.compute_consistent_loss(logits, hier_matrixs, device)

        total_loss = classification_loss + 0.5 * consistent_loss

        return classification_loss, each_classification_loss, consistent_loss, each_consistent_loss, total_loss