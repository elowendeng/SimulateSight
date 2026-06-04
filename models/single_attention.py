# models/single_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * attn


class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention(in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 2, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_out = self.channel_attn(x)
        spatial_out = self.spatial_attn(x)
        concat = torch.cat([channel_out, spatial_out], dim=1)
        gate = self.gate(concat)
        output = gate[:, 0:1] * channel_out + gate[:, 1:2] * spatial_out
        return output


class MultiAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.channel_weight = nn.Parameter(torch.tensor(0.34))
        self.spatial_weight = nn.Parameter(torch.tensor(0.33))
        self.pixel_weight = nn.Parameter(torch.tensor(0.33))
        
        self.channel_attn = ChannelAttention(in_channels, reduction)
        
        self.spatial_attn = SpatialAttention(in_channels)
        
        self.pixel_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 3, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.residual = nn.Identity()
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        identity = x
        channel_out = self.channel_attn(x)
        spatial_out = self.spatial_attn(x)
        
        pixel_weights = self.pixel_attn(x)
        pixel_out = x * pixel_weights
        
        cw = torch.sigmoid(self.channel_weight)
        sw = torch.sigmoid(self.spatial_weight)
        pw = torch.sigmoid(self.pixel_weight)
        total = cw + sw + pw + 1e-8
        cw, sw, pw = cw/total, sw/total, pw/total
        
        weighted_fusion = cw * channel_out + sw * spatial_out + pw * pixel_out
        
        concat_for_gate = torch.cat([channel_out, spatial_out, pixel_out], dim=1)
        gate_weights = self.gate(concat_for_gate)
        
        gated_fusion = gate_weights[:, 0:1] * channel_out + \
                       gate_weights[:, 1:2] * spatial_out + \
                       gate_weights[:, 2:3] * pixel_out
        
        alpha = torch.sigmoid(self.alpha)
        output = alpha * weighted_fusion + (1 - alpha) * gated_fusion
        
        concat_output = torch.cat([output, channel_out, spatial_out], dim=1)
        output = self.fusion(concat_output)
        
        output = output + self.residual(identity)
        return output


class ImprovedSpatialAttention(nn.Module):

    def __init__(self, in_channels, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 5, padding=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 7, padding=3),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 4 * 3, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.smooth = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
    def forward(self, x):
        identity = x

        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)

        concat = torch.cat([feat1, feat2, feat3], dim=1)
        attn = self.fusion(concat)

        attn = self.smooth(attn)
        output = x * attn
        if self.use_residual:
            residual_weight = torch.sigmoid(self.residual_weight)
            output = output + residual_weight * identity
        
        return output


class StdPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        std = torch.std(x, dim=[2, 3], keepdim=True)
        std = std / (std.max() + 1e-8)
        return std


class ImprovedChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, use_1d_conv=True):
        super().__init__()
        
        self.use_1d_conv = use_1d_conv
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.std_pool = StdPool2d()
        
        if use_1d_conv:
            kernel_size = max(3, in_channels // reduction // 2)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            
            self.conv1d = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.Sigmoid()
            )
            self.shared_mlp = None
        else:
            self.shared_mlp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
            )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        identity = x

        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        std_feat = self.std_pool(x)
        
        if self.use_1d_conv:
            avg_out = self._conv1d_attention(avg_feat)
            max_out = self._conv1d_attention(max_feat)
            std_out = self._conv1d_attention(std_feat)
        else:
            avg_out = self.shared_mlp(avg_feat)
            max_out = self.shared_mlp(max_feat)
            std_out = self.shared_mlp(std_feat)

        concat = torch.cat([avg_out, max_out, std_out], dim=1)
        attn = self.gate(concat)

        output = x * attn
        residual_weight = torch.sigmoid(self.residual_weight)
        output = output + residual_weight * identity
        
        return output
    
    def _conv1d_attention(self, feat):
        # feat shape: [B, C, 1, 1]
        B, C, H, W = feat.shape
        feat = feat.view(B, C)  # [B, C]
        feat = feat.unsqueeze(1)  # [B, 1, C]
        attn = self.conv1d(feat)  # [B, 1, C]
        attn = attn.view(B, 1, C, 1, 1)  # [B, 1, C, 1, 1]
        attn = attn.squeeze(1)  # [B, C, 1, 1]
        return attn


class ImprovedDualAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.channel_attn = ImprovedChannelAttention(in_channels, reduction)
        self.spatial_attn = ImprovedSpatialAttention(in_channels)

        self.co_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 3, padding=1),
            nn.Sigmoid()
        )

        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 2, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        identity = x
        channel_out = self.channel_attn(x)
        spatial_out = self.spatial_attn(x)
        concat_co = torch.cat([channel_out, spatial_out], dim=1)
        co_weights = self.co_attention(concat_co)

        enhanced_channel = channel_out * co_weights[:, 0:1]
        enhanced_spatial = spatial_out * co_weights[:, 1:2]

        gate_weights = self.dynamic_gate(x)
        dynamic_channel_weight = gate_weights[:, 0:1]
        dynamic_spatial_weight = gate_weights[:, 1:2]

        base_fusion = dynamic_channel_weight * enhanced_channel + \
                      dynamic_spatial_weight * enhanced_spatial
        max_fusion = torch.max(enhanced_channel, enhanced_spatial)
        avg_fusion = (enhanced_channel + enhanced_spatial) / 2
        alpha = torch.sigmoid(self.alpha)
        output = alpha * base_fusion + (1 - alpha) * max_fusion
        output = output + 0.3 * avg_fusion

        concat_refine = torch.cat([output, identity], dim=1)
        output = self.refine(concat_refine)
        output = output + self.residual_conv(identity)
        
        return output


class ImprovedMultiAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.weight1 = nn.Parameter(torch.tensor(0.34))
        self.weight2 = nn.Parameter(torch.tensor(0.33))
        self.weight3 = nn.Parameter(torch.tensor(0.33))
        
        self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.pixel_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.33))
        self.beta = nn.Parameter(torch.tensor(0.33))
        
    def forward(self, x):
        identity = x

        channel_weight = self.channel_attn(x)
        channel_out = x * channel_weight
        
        spatial_weight = self.spatial_attn(x)
        spatial_out = x * spatial_weight
        
        pixel_weight = self.pixel_attn(x)
        pixel_out = x * pixel_weight

        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        w3 = torch.sigmoid(self.weight3)
        total = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total, w2/total, w3/total
        
        weighted_fusion = w1 * channel_out + w2 * spatial_out + w3 * pixel_out

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        output = weighted_fusion + alpha * channel_out + beta * spatial_out
        output = output + (1 - alpha - beta) * pixel_out

        concat_refine = torch.cat([output, identity], dim=1)
        output = self.refine(concat_refine)

        output = output + self.residual_conv(identity)
        
        return output


def get_attention_module(attn_type, in_channels, use_improved=True):
    if use_improved:
        if attn_type == 'spatial':
            return ImprovedSpatialAttention(in_channels)
        elif attn_type == 'channel':
            return ImprovedChannelAttention(in_channels)
        elif attn_type == 'dual':
            return ImprovedDualAttention(in_channels)
        elif attn_type == 'multi':
            return ImprovedMultiAttention(in_channels)
        else:
            return nn.Identity()
    else:
        if attn_type == 'spatial':
            return SpatialAttention(in_channels)
        elif attn_type == 'channel':
            return ChannelAttention(in_channels)
        elif attn_type == 'dual':
            return DualAttention(in_channels)
        elif attn_type == 'multi':
            return MultiAttention(in_channels)
        else:
            return nn.Identity()