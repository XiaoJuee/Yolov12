import torch
import torch.nn as nn

class ECAAttention(nn.Module):
    """高效通道注意力 (CVPR 2020) - 修正版"""
    def __init__(self, c1, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.kernel_size = kernel_size
        
        # 根据通道数设置卷积
        if kernel_size > 1:
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                                  padding=(kernel_size-1)//2, bias=False)
        else:
            self.conv = nn.Identity()
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x)  # [b, c, 1, 1]
        
        if self.kernel_size > 1:
            # 处理为1D卷积输入
            y = y.view(b, c, 1)
            y = y.transpose(1, 2)  # [b, 1, c]
            y = self.conv(y)  # 1D卷积
            y = y.transpose(1, 2)  # [b, c, 1]
            y = y.view(b, c, 1, 1)
        else:
            # 当kernel_size=1时直接使用
            y = y
        
        return x * self.sigmoid(y)

class SimAM(nn.Module):
    """无参注意力机制 (CVPR 2021)"""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activ = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        # 计算能量函数
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mu).pow(2)
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activ(y)

class CBAM(nn.Module):
    """卷积块注意力模块 (ECCV 2018)"""
    def __init__(self, c1, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(c1 // reduction_ratio, c1, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        sa = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa)
        
        return x_ca * sa