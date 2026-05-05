import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNtoSwinAdapter(nn.Module):
    def __init__(self, pool_layer=False, kernel_size=2):
        self.pool_layer = pool_layer
        self.kernel_size = kernel_size
        self.build_layers()
    def build_layers(self):
        self.pooling_layer = nn.AvgPool2d(kernel_size=self.kernel_size)
    def forward(self, x):
        if self.pool_layer:
            x = self.pooling_layer(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
class FusionBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.normA = nn.BatchNorm2d(2048)
        self.normB = nn.BatchNorm2d(1024)

        # learnable scaling
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, A, B):
        A = self.normA(A)
        B = self.normB(B)

        A = self.alpha * A
        B = self.beta * B

        return torch.cat([A, B], dim=1)  # 3072 channels