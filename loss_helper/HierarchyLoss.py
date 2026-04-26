import torch.nn as nn

class HierarchyLoss(nn.Module):
    def __init__(self):
        super(HierarchyLoss, self).__init__()
        """hierarchy={'class': 100, 'family': 47, 'order': 18}
        """
        # self.hierarchical_class = list(hierarchy.values())
        self.CrossEntropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss0 = self.CrossEntropy(inputs[0], targets[0])
        loss1 = self.CrossEntropy(inputs[1], targets[1])
        loss2 = self.CrossEntropy(inputs[2], targets[2])
        loss3 = self.CrossEntropy(inputs[3], targets[3])
        loss = loss0 + loss1 + loss2 + loss3

        return loss