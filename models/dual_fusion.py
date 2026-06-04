# models/dual_fusion.py

import torch
import torch.nn as nn


class SpatialAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        attn = self.conv(concat)
        fused = attn[:, 0:1] * x1 + attn[:, 1:2] * x2
        output = self.fusion(torch.cat([fused, concat], dim=1))
        return output


class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        avg_pool = self.avg_pool(concat)
        max_pool = self.max_pool(concat)
        attn = self.fc(avg_pool + max_pool)
        attended = concat * attn
        output = self.fusion(attended)
        return output


class DualAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.channel_attn = ChannelAttentionFusion(in_channels)
        self.spatial_attn = SpatialAttentionFusion(in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 2, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        channel_out = self.channel_attn(x1, x2)
        spatial_out = self.spatial_attn(x1, x2)
        concat = torch.cat([channel_out, spatial_out], dim=1)
        gate = self.gate(concat)
        output = gate[:, 0:1] * channel_out + gate[:, 1:2] * spatial_out
        return output


class MultiAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x1, x2):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1/total, w2/total
        weighted_fusion = w1 * x1 + w2 * x2
        
        concat = torch.cat([x1, x2], dim=1)
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x1 + spatial_weights[:, 1:2] * x2
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        channel_fusion = channel_fusion[:, :x1.shape[1], :, :] + \
                         channel_fusion[:, x1.shape[1]:, :, :]
        channel_fusion = channel_fusion / 2
        
        gate_input = torch.cat([weighted_fusion, spatial_fusion, channel_fusion], dim=1)
        gate_weights = self.gate(torch.cat([x1, x2], dim=1))
        
        output = gate_weights[:, 0:1] * weighted_fusion + \
                 gate_weights[:, 1:2] * spatial_fusion
        
        output = output + 0.5 * channel_fusion
        
        output = self.fusion(output)
        
        output = output + self.residual(x1)
        return output


class ImprovedDualAttentionFusion(nn.Module):

    def __init__(self, in_channels=256):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)

        self.channel_attn = ChannelAttentionFusion(in_channels)
        self.spatial_attn = SpatialAttentionFusion(in_channels)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 2, 3, padding=1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1 / total, w2 / total
        weighted_fusion = w1 * x1 + w2 * x2

        channel_out = self.channel_attn(x1, x2)
        spatial_out = self.spatial_attn(x1, x2)
        concat = torch.cat([channel_out, spatial_out], dim=1)
        gate = self.gate(concat)
        attn_fusion = gate[:, 0:1] * channel_out + gate[:, 1:2] * spatial_out

        residual = self.residual_conv(x1)

        alpha = torch.sigmoid(self.alpha)
        output = weighted_fusion + alpha * attn_fusion + (1 - alpha) * residual

        output = self.final_conv(output)

        return output


class ImprovedSpatialAttentionFusion(nn.Module):

    def __init__(self, in_channels=256):
        super().__init__()

        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 3, padding=1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x1, x2):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1 / total, w2 / total

        concat = torch.cat([x1, x2], dim=1)
        attn = self.conv(concat)
        fused = attn[:, 0:1] * x1 + attn[:, 1:2] * x2

        weighted = w1 * x1 + w2 * x2
        output = self.fusion(torch.cat([fused + weighted, concat], dim=1))
        output = output + self.residual(x1)

        return output


class ImprovedChannelAttentionFusion(nn.Module):

    def __init__(self, in_channels=256):
        super().__init__()

        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x1, x2):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1 / total, w2 / total

        concat = torch.cat([x1, x2], dim=1)
        avg_pool = self.avg_pool(concat)
        max_pool = self.max_pool(concat)
        attn = self.fc(avg_pool + max_pool)
        attended = concat * attn

        weighted = w1 * x1 + w2 * x2
        output = self.fusion(attended)
        output = output + self.residual(weighted)

        return output


class ImprovedMultiAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))
        
        self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x1, x2):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1/total, w2/total
        weighted_fusion = w1 * x1 + w2 * x2
        
        concat = torch.cat([x1, x2], dim=1)
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        c1 = channel_fusion[:, :x1.shape[1], :, :]
        c2 = channel_fusion[:, x1.shape[1]:, :, :]
        channel_fusion = (c1 + c2) / 2
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x1 + spatial_weights[:, 1:2] * x2
        
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        output = weighted_fusion + alpha * channel_fusion + beta * spatial_fusion
        output = self.final_conv(output)
        output = output + self.residual_conv(x1)
        return output


def get_dual_fusion_module(attn_type, in_channels, use_improved=True):
    if use_improved:
        if attn_type == 'spatial':
            return ImprovedSpatialAttentionFusion(in_channels)
        elif attn_type == 'channel':
            return ImprovedChannelAttentionFusion(in_channels)
        elif attn_type == 'dual':
            return ImprovedDualAttentionFusion(in_channels)
        elif attn_type == 'multi':
            return ImprovedMultiAttentionFusion(in_channels)
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
    else:
        if attn_type == 'spatial':
            return SpatialAttentionFusion(in_channels)
        elif attn_type == 'channel':
            return ChannelAttentionFusion(in_channels)
        elif attn_type == 'dual':
            return DualAttentionFusion(in_channels)
        elif attn_type == 'multi':
            return MultiAttentionFusion(in_channels)
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )