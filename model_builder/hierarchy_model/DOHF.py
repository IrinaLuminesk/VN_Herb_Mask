import torch.nn as nn
import torch.nn.functional as F
class DOHF(nn.Module):
    def __init__(self, add_lambda=0.8):
        super().__init__()
        self.add_lambda = add_lambda
        self.eps = 1e-6

    def forward(self, shallow, deep):
        if shallow is None:
            return deep

        # projection of deep onto shallow
        proj = (deep * shallow).sum(dim=1, keepdim=True) * shallow
        norm_sq = (shallow ** 2).sum(dim=1, keepdim=True) + self.eps
        proj = proj / norm_sq

        # orthogonal component
        orth = deep - proj

        # enhance orthogonal part
        deep = deep + self.add_lambda * orth

        return F.normalize(deep, dim=-1)
