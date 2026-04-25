import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class AttentionDohfNeck2(nn.Module):

    def __init__(self, M=32, res_channels=2048, pooling_mode='GAP', add_lambda=0.8):
        super(AttentionDohfNeck2, self).__init__()
        self.M = M
        self.base_channels = res_channels
        self.out_channels = M * res_channels
        self.conv = BasicConv2d(res_channels, self.M, kernel_size=1)

        self.pooling = self.build_pooling(pooling_mode)
        self.EPSILON = 1e-6

        self.add_lambda = add_lambda

    def build_pooling(self, pooling_mode):
        if pooling_mode == "GAP":
            return None
        elif pooling_mode == "GMP":
            return nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("Unknown pooling mode: {}".format(pooling_mode))

    def bilinear_attention_pooling(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=False)
            # attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pooling is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pooling(features * attentions[:, i:i + 1, ...]).view(B, -1)  # (B, C)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)  # (B, M * C)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        return feature_matrix

    def forward(self, x):
        attention_maps = self.conv(x)
        feature_matrix = self.bilinear_attention_pooling(x, attention_maps)
        return feature_matrix, attention_maps  # (B, M * C), (B, M, AH, AW)

    # CHOF
    def dohf(self, shallow_hiera, deep_hiera):
        """
        from shallow to deep: order, family, genus, class
        shallow_hiera: N, M*C
        deep_hiera: N, M*C
        return
        """
        if shallow_hiera==None:
            return deep_hiera, deep_hiera
        N1, MC1 = shallow_hiera.shape
        M1 = MC1//self.base_channels
        shallow_hiera_mean = shallow_hiera.reshape(N1, M1, self.base_channels)  # N,M1*C -> N,M1,C
        shallow_hiera_mean = shallow_hiera_mean.mean(dim=1)  # N, C

        N2, MC2 = deep_hiera.shape
        M2 = MC2//self.base_channels
        deep_hiera_dohf = deep_hiera.reshape(N2, M2, self.base_channels)  # N,M2*C -> N,M2,C
        deep_hiera_dohf = deep_hiera_dohf.permute(0, 2, 1).contiguous()  # N,M2,C -> N,C,M2

        projection = torch.bmm(shallow_hiera_mean.unsqueeze(1), deep_hiera_dohf)  # N, 1, M2
        projection = torch.bmm(shallow_hiera_mean.unsqueeze(2), projection)  # N, C, M2
        shallow_hiera_norm = torch.norm(shallow_hiera_mean, p=2, dim=1)  # N
        projection = projection / (shallow_hiera_norm * shallow_hiera_norm).view(-1, 1, 1)  # N, C, M2

        orthogonal_comp = deep_hiera_dohf - projection
        deep_hiera_dohf = deep_hiera_dohf + self.add_lambda * orthogonal_comp  # N, C, M2
        deep_hiera_dohf = deep_hiera_dohf.permute(0, 2, 1).contiguous()  # N, C, M2 -> N,M2,C
        deep_hiera_dohf = deep_hiera_dohf.reshape(N2, -1)  # N,M2,C -> N, MC2
        # l2 normalization along dimension M2 and C
        deep_hiera_dohf = F.normalize(deep_hiera_dohf, dim=-1)
        return deep_hiera, deep_hiera_dohf

Att_MAP = {'class': 32, 'genus': 16, 'family': 8, 'order': 4}
class GINN(nn.Module):
    """ Constructor
        Args:
            hierarchy: {'species':100, 'family':47, 'order':18}
            use_attention: if use attentions?
    """
    def __init__(self, hierarchy, use_attention=True):   
        super(GINN, self).__init__()
        self.hierarchy = hierarchy
        self.hier_names = list(hierarchy.keys()) #Lấy các key của dict theo cấp thấp tới cao
        self.hierarchical_depth = len(hierarchy) #Lấy độ sâu của dict
        self.use_attention = use_attention
        self.build_layers()
    def build_layers(self):
        resnet_weights = ResNet50_Weights.DEFAULT
        backbone_model = resnet50(weights=resnet_weights)

        self.feature_embedding = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
        )

        self.hier_branch = {}
        # species, (genera), family, order
        for hier in self.hier_names:
            hier_stage = nn.Sequential(
                nn.Conv2d(1024, 1536, 3, 1, 1, bias=False),
                nn.BatchNorm2d(1536),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(1536, 2048, 3, 1, 1, bias=False),
                nn.BatchNorm2d(2048)
            )
            self.hier_branch[hier] = hier_stage

        self.hier_neck = {}
        for hier in self.hier_names:
            hier_stage = AttentionDohfNeck2(M=Att_MAP[hier])
            self.hier_neck[hier] = hier_stage

        self.hier_classifyhead = {}
        for hier in self.hier_names:
            hier_stage = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(1024, int(self.hierarchy[hier])),
            )
            self.hier_classifyhead[hier] = hier_stage
    def forward(self, x):
        batch_size = x.size(0)
        # trunk
        x = self.feature_embedding(x)  # (B,1024,4,8)
        # branch
        multih_fmap = {}
        for hier in self.hier_names:
            hier_x = x
            hier_x = self.hier_branch[hier](hier_x)
            multih_fmap[hier] = hier_x   # (B,2048,2,8)

        multih_fmatrixs, multih_scores = [], []

        shallow_feature_matrix = None
        for hier in reversed(self.hier_names):
        #for hier in self.hier_names:
            feature_matrix, _ = self.hier_neck[hier](multih_fmap[hier])
            # aggregate single hierarchy feature
            # multih_fmatrixs[hier] = feature_matrix
            shallow_feature_matrix, feature_matrix = self.hier_neck[hier].dohf(shallow_feature_matrix, feature_matrix)
            scores = self.hier_classifyhead[hier](feature_matrix)
            # aggregate dohf hierarchy feature
            multih_fmatrixs.append(feature_matrix)
            multih_scores.append(scores)
        return multih_scores[::-1]

