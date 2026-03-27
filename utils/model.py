import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torchvision import models


class SharedEncoder(torch_nn.Module):
    def __init__(self, in_channels=3, backbone='resnet18', pretrained=False, 
                 pretrained_path=None, dropout=0.3):
        """
        Args:
            in_channels: 输入通道数
            backbone: ResNet 类型 ('resnet18', 'resnet34', 'resnet50')
            pretrained: 是否使用 torchvision 自动下载的预训练权重
            pretrained_path: 本地预训练权重路径（优先级高于 pretrained)
            dropout: Dropout 概率
        """
        super(SharedEncoder, self).__init__()
        
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=None)
            self.out_channels = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=None)
            self.out_channels = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=None)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if pretrained_path and pretrained_path.strip():
            print(f"Loading pretrained weights from local path: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = resnet.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"  Unexpected keys (ignored): {len(unexpected_keys)} keys")
            print(f"  Loaded pretrained weights successfully")
        elif pretrained:
            print(f"Loading pretrained {backbone} from torchvision")
            if backbone == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT
            elif backbone == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT
            elif backbone == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet18(weights=weights) if backbone == 'resnet18' else \
                     models.resnet34(weights=weights) if backbone == 'resnet34' else \
                     models.resnet50(weights=weights)
        
        if in_channels != 3:
            first_conv = resnet.conv1
            resnet.conv1 = torch_nn.Conv2d(
                in_channels, 
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias
            )
            if (pretrained or pretrained_path is not None) and in_channels < 3:
                with torch.no_grad():
                    resnet.conv1.weight[:] = first_conv.weight[:, :in_channels, :, :]
        
        # 使用 ResNet 的前四层（去掉最后的全局池化和全连接层）
        # 输出 stride = 32，即输入 96x96 会输出 3x3 的特征图
        self.encoder = torch_nn.Sequential(
            resnet.conv1,    # 1/2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # 1/4
            resnet.layer1,   # 1/4
            resnet.layer2,   # 1/8
            resnet.layer3,   # 1/16
            resnet.layer4,   # 1/32
        )
        
        self.dropout = torch_nn.Dropout2d(p=dropout) if dropout > 0 else None

        for param in self.encoder[0:6].parameters(): # 冻结 conv1 到 layer2
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class CrossAttentionAlignment(torch_nn.Module):
    def __init__(self, channels, num_heads=8):
        """
        Args:
            channels: 特征图的通道数 (ResNet18/34 为 512, ResNet50 为 2048)
            num_heads: 多头注意力的头数，要求 channels 必须能被 num_heads 整除
        """
        super(CrossAttentionAlignment, self).__init__()

        self.multihead_attn = torch_nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.gamma = torch_nn.Parameter(torch.zeros(1)) 

    def forward(self, feat_A, feat_B):
        B, C, H, W = feat_A.size()
        
        query = feat_A.view(B, C, -1).permute(0, 2, 1)
        key   = feat_B.view(B, C, -1).permute(0, 2, 1)
        value = feat_B.view(B, C, -1).permute(0, 2, 1)
        
        attn_output, _ = self.multihead_attn(query, key, value)
        
        aligned_B = attn_output.permute(0, 2, 1).view(B, C, H, W)
        
        out = self.gamma * aligned_B + feat_B
        
        return out



class HomBlock(torch_nn.Module):
    #差值
    def __init__(self, channels, num_heads=8):
        super(HomBlock, self).__init__()
        self.multihead_attn = torch_nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )

    def forward(self, query_feat, key_value_feat):
        B, C, H, W = query_feat.size()
        
        q = query_feat.view(B, C, -1).permute(0, 2, 1)
        k = key_value_feat.view(B, C, -1).permute(0, 2, 1)
        v = key_value_feat.view(B, C, -1).permute(0, 2, 1)
        
        attn_output, _ = self.multihead_attn(q, k, v)
        aligned_feat = attn_output.permute(0, 2, 1).view(B, C, H, W)
        
        diff = torch.abs(query_feat - aligned_feat)
        return diff


class SiameseAnomalyNet(torch_nn.Module):
    def __init__(self, in_channels=3, backbone='resnet18', pretrained=False, 
                 pretrained_path=None, dropout=0.3):
        super(SiameseAnomalyNet, self).__init__()
        
        self.encoder = SharedEncoder(in_channels, backbone, pretrained, pretrained_path, dropout)
        
        feat_channels = self.encoder.out_channels
        
        self.aligner = HomBlock(channels=feat_channels, num_heads=8)
        
        self.W_hom = torch_nn.Sequential(
            torch_nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1),
            torch_nn.BatchNorm2d(feat_channels),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout2d(p=dropout),

            torch_nn.Conv2d(feat_channels, feat_channels // 4, kernel_size=1),
            torch_nn.BatchNorm2d(feat_channels // 4),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout2d(p=dropout/2),
            torch_nn.Conv2d(feat_channels // 4, 1, kernel_size=1)
        )
        
        self.gap = torch_nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = torch_nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, img_A, mask_A, img_B, mask_B):
        x_A = img_A * mask_A
        x_B = img_B * mask_B

        feat_A = self.encoder(x_A)
        feat_B = self.encoder(x_B)
        
        R_1 = self.aligner(query_feat=feat_A, key_value_feat=feat_B)
        
        R_2 = self.aligner(query_feat=feat_B, key_value_feat=feat_A)
        
        concat_diff = torch.cat([R_1, R_2], dim=1) 
        
        heatmap = self.W_hom(concat_diff)  
        
        pooled_avg = self.gap(heatmap)
        pooled_max = self.gmp(heatmap)
        pooled = pooled_avg + pooled_max
        
        logits = pooled.view(pooled.size(0), -1) 

        return logits, heatmap