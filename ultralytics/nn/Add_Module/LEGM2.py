import torch.nn as nn
from ultralytics.nn.modules import Conv

class LEGM(nn.Module):
    """局部特征嵌入全局特征提取模块"""
    def __init__(self, c1, reduction=16):
        super().__init__()
        # 多尺度特征融合（示例简化版）
        self.conv_low = Conv(c1, c1//reduction, 1)  # 局部特征压缩
        self.conv_high = Conv(c1, c1, 3)  # 全局特征提取
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(c1//reduction, c1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_low = self.conv_low(x)  # 局部特征
        x_high = self.conv_high(x)  # 全局特征
        attn = self.attention(x_high)  # 注意力权重
        return x_low * attn + x_high  # 融合局部与全局特征