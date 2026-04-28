import torch.nn as nn
import torch
class HierarchyLoss(nn.Module):
    def __init__(self):
        super(HierarchyLoss, self).__init__()
        """hierarchy={'class': 100, 'family': 47, 'order': 18}
        """
        # self.hierarchical_class = list(hierarchy.values())
        self.CrossEntropy = nn.CrossEntropyLoss()
        # {'species':200 , 'genus': 125, 'family': 36, 'order': 13}
        self.S2G = self.build_mapping(200, 125, species_to_genus, device)
        self.G2F = self.build_mapping(num_genus, num_family, genus_to_family, device)
        self.F2O = self.build_mapping(num_family, num_order, family_to_order, device)

    def build_mapping(self, num_fine, num_coarse, mapping_dict, device):
        M = torch.zeros(num_fine, num_coarse, device=device)
        
        for fine, coarse in mapping_dict.items():
            M[fine, coarse] = 1.0

        # normalize rows
        M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
        return M

    def forward(self, outputs, targets):
        loss0 = self.CrossEntropy(outputs["species"], targets[:, 0])
        loss1 = self.CrossEntropy(outputs["genus"], targets[:, 1])
        loss2 = self.CrossEntropy(outputs["family"], targets[:, 2])
        loss3 = self.CrossEntropy(outputs["order"], targets[:, 3])
        loss = loss0 + loss1 + loss2 + loss3

        return loss