import torch
import torch.nn as nn
import torch.nn.functional as F

class Spatial_Attention_Module(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output)
        return output 

class Channel_Attention_Module(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output)
        return output

class CBAM(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.channels = channels
        self.r = r
        self.spatial_attention_module = Spatial_Attention_Module(bias=False)
        self.channel_attention_module = Channel_Attention_Module(channels=self.channels, r=self.r)
    def forward(self, x):
        output = self.channel_attention_module(x)
        output = self.spatial_attention_module(output)
        return output + x
    

class BidirectionalAttentionModule(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()

        self.channels = channels
        self.r = r
        self.spatial_attention_module = Spatial_Attention_Module(bias=False)
        self.channel_attention_module = Channel_Attention_Module(channels=self.channels, r=self.r)

        self.fusion = nn.Conv2d(channels * 2, self.channels, kernel_size=1)

    def forward(self, f1, f2):
        """
        f1: original feature map
        f2: augmented feature map
        """

        # ---- Bidirectional Spatial Attention ----
        s1 = self.spatial_attention_module(f2)   # attention from f2 to f1
        s2 = self.spatial_attention_module(f1)   # attention from f1 to f2

        f1 = f1 * s1
        f2 = f2 * s2

        # ---- Channel Attention ----
        c1 = self.channel_attention_module(f1)
        c2 = self.channel_attention_module(f2)

        f1 = f1 * c1
        f2 = f2 * c2

        # ---- Feature Fusion ----
        fused = torch.cat([f1, f2], dim=1)
        fused = self.fusion(fused)

        return fused