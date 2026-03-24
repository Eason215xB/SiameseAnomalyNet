import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torchvision import models


class SharedEncoder(torch_nn.Module):
    """
    孪生编码器：流 A 和 流 B 共享这个网络。
    使用 ResNet18/34/50 作为 backbone 提取染色体的深度语义特征。
    """
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
        
        # 加载预训练的 ResNet
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
        
        # 加载预训练权重（优先使用本地路径）
        if pretrained_path and pretrained_path.strip():
            # 从本地路径加载
            print(f"Loading pretrained weights from local path: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # 处理可能的 'module.' 前缀（来自 DDP 训练）
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # 尝试加载，忽略不匹配的层（如 fc 层）
            missing_keys, unexpected_keys = resnet.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"  Unexpected keys (ignored): {len(unexpected_keys)} keys")
            print(f"  Loaded pretrained weights successfully")
        elif pretrained:
            # 从 torchvision 下载
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
        
        # 适配输入通道数（ResNet 默认是 3 通道）
        if in_channels != 3:
            # 替换第一层卷积以适配不同的输入通道
            first_conv = resnet.conv1
            resnet.conv1 = torch_nn.Conv2d(
                in_channels, 
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias
            )
            # 可选：用原 3 通道权重的平均来初始化新通道
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
        
        # Dropout 用于正则化
        self.dropout = torch_nn.Dropout2d(p=dropout) if dropout > 0 else None

        for param in self.encoder[0:6].parameters(): # 冻结 conv1 到 layer2
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class CrossAttentionAlignment(torch_nn.Module):
    """
    多头交叉注意力对齐模块 (Multi-Head Cross Attention)：
    用于克服染色体自由弯曲带来的空间不对齐。
    以健康的染色体 A 为基准 (Query)，去异常染色体 B 中寻找相似的结构 (Key, Value) 并拼凑对齐。
    """
    def __init__(self, channels, num_heads=8):
        """
        Args:
            channels: 特征图的通道数 (ResNet18/34 为 512, ResNet50 为 2048)
            num_heads: 多头注意力的头数，要求 channels 必须能被 num_heads 整除
        """
        super(CrossAttentionAlignment, self).__init__()
        
        # 使用 PyTorch 官方高度优化的多头注意力机制
        # batch_first=True 允许输入张量形状为 (Batch, Sequence_Length, Features)
        self.multihead_attn = torch_nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 可学习的残差缩放因子
        self.gamma = torch_nn.Parameter(torch.zeros(1)) 

    def forward(self, feat_A, feat_B):
        B, C, H, W = feat_A.size()
        
        # 1. 展平空间维度，将特征图转化为 Transformer 标准的序列 (Token) 输入
        # 形状转换: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        # 这里的 H*W 就是序列长度 (Sequence Length)，C 是特征维度 (Embedding Dim)
        query = feat_A.view(B, C, -1).permute(0, 2, 1)
        key   = feat_B.view(B, C, -1).permute(0, 2, 1)
        value = feat_B.view(B, C, -1).permute(0, 2, 1)
        
        # 2. 计算多头交叉注意力
        # query 是基准 A，key 和 value 是目标 B
        # attn_output 形状: (B, H*W, C)
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # 3. 将对齐后的序列特征还原回 2D 空间特征图格式
        # 形状转换: (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        aligned_B = attn_output.permute(0, 2, 1).view(B, C, H, W)
        
        # 4. 残差连接 (在 B 的原始特征上，叠加注意力对齐后的修正量)
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
        # 移除华而不实的 W_diff，我们强制网络只看差异！

    def forward(self, query_feat, key_value_feat):
        B, C, H, W = query_feat.size()
        
        q = query_feat.view(B, C, -1).permute(0, 2, 1)
        k = key_value_feat.view(B, C, -1).permute(0, 2, 1)
        v = key_value_feat.view(B, C, -1).permute(0, 2, 1)
        
        attn_output, _ = self.multihead_attn(q, k, v)
        aligned_feat = attn_output.permute(0, 2, 1).view(B, C, H, W)
        
        # 核心修复：强制计算“绝对差值”！这是避免过拟合的定海神针
        diff = torch.abs(query_feat - aligned_feat)
        return diff


class SiameseAnomalyNet(torch_nn.Module):
    """
    完整的双流异常检测网络 (弱监督定位版) - ResNet Backbone
    """
    def __init__(self, in_channels=3, backbone='resnet18', pretrained=False, 
                 pretrained_path=None, dropout=0.3):
        super(SiameseAnomalyNet, self).__init__()
        
        # 1. 孪生特征提取器 (共享参数) - ResNet Backbone
        self.encoder = SharedEncoder(in_channels, backbone, pretrained, pretrained_path, dropout)
        
        feat_channels = self.encoder.out_channels
        
        # 2. 对齐模块
        self.aligner = HomBlock(channels=feat_channels, num_heads=8)
        
        # 3. 异常热力图生成器 (将差异特征降维到 1 个通道，即二维 Heatmap)
        self.W_hom = torch_nn.Sequential(
            torch_nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1),
            torch_nn.BatchNorm2d(feat_channels),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout2d(p=dropout),
            torch_nn.Conv2d(feat_channels, feat_channels // 4, kernel_size=3, padding=1),
            torch_nn.BatchNorm2d(feat_channels // 4),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout2d(p=dropout/2),
            torch_nn.Conv2d(feat_channels // 4, 1, kernel_size=1)  # 最终输出单通道异常 Heatmap
        )
        
        # 4. 全局平均池化 (GAP) 用于图像级分类
        self.gap = torch_nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img_A, mask_A, img_B, mask_B):
        # --- 第一步：背景抑制 ---
        x_A = img_A * mask_A
        x_B = img_B * mask_B
        
        # --- 第二步：双流特征提取 ---
        feat_A = self.encoder(x_A)
        feat_B = self.encoder(x_B)
        
        # --- 第三步：双向多头注意力对齐 (Bidirectional Alignment) ---
        # ˆR_{i,1} : A作为基准(Query)，去B中找对应特征(Key/Value)
        R_1 = self.aligner(query_feat=feat_A, key_value_feat=feat_B)
        
        # ˆR_{i,2} : B作为基准(Query)，去A中找对应特征(Key/Value)
        R_2 = self.aligner(query_feat=feat_B, key_value_feat=feat_A)
        
        # --- 第四步：整合双向差异特征 (对应论文的 ˆR_{i,1} ⊕ ˆR_{i,2}) ---
        # dim=1 表示在通道层面上进行 Concat (⊕)
        concat_diff = torch.cat([R_1, R_2], dim=1) 
        
        # --- 第五步：特征融合与降维 (对应论文的 W_hom) ---
        # 融合后直接生成单通道的高亮异常图
        heatmap = self.W_hom(concat_diff) 
        
        # --- 第六步：特征展平分类 (对应论文的 Flat) ---
        pooled = self.gap(heatmap) 
        logits = pooled.view(pooled.size(0), -1) 

        return logits, heatmap

# ================= 测试运行与使用示例 =================
if __name__ == "__main__":
    B, C, H, W = 4, 3, 96, 96
    
    dummy_img_A = torch.rand(B, C, H, W)
    dummy_mask_A = torch.ones(B, C, H, W)
    
    dummy_img_B = torch.rand(B, C, H, W)
    dummy_mask_B = torch.ones(B, C, H, W)
    
    # 测试不同的 ResNet backbone
    for backbone in ['resnet18', 'resnet34', 'resnet50']:
        print(f"\n=== 测试 {backbone} ===")
        model = SiameseAnomalyNet(in_channels=C, backbone=backbone, pretrained=False, dropout=0.3)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 前向传播
        logits, heatmap = model(dummy_img_A, dummy_mask_A, dummy_img_B, dummy_mask_B)
        prob = torch.sigmoid(logits)
        
        print(f"输入: {dummy_img_A.shape}")
        print(f"特征图输出: {heatmap.shape}")
        print(f"最终分类概率 (Probability): {prob.shape} -> 代表这对染色体异常的概率")
        print(f"异常热力图 (Anomaly Heatmap): {heatmap.shape} -> 可直接叠加回原图显示高光异常区域")