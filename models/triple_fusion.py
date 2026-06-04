# models/triple_fusion.py

import torch
import torch.nn as nn


class TripleAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        self.weight1 = nn.Parameter(torch.tensor(0.33))
        self.weight2 = nn.Parameter(torch.tensor(0.33))
        self.weight3 = nn.Parameter(torch.tensor(0.34))
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 3, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 3, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2, x3):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        w3 = torch.sigmoid(self.weight3)
        total = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total, w2/total, w3/total
        weighted_fusion = w1 * x1 + w2 * x2 + w3 * x3
        
        concat = torch.cat([x1, x2, x3], dim=1)
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x1 + \
                         spatial_weights[:, 1:2] * x2 + \
                         spatial_weights[:, 2:3] * x3
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        channel_fusion = channel_fusion[:, :x1.shape[1], :, :] + \
                         channel_fusion[:, x1.shape[1]:x1.shape[1]*2, :, :] + \
                         channel_fusion[:, x1.shape[1]*2:, :, :]
        channel_fusion = channel_fusion / 3
        
        gate_weights = self.gate(concat)
        output = gate_weights[:, 0:1] * weighted_fusion + \
                 gate_weights[:, 1:2] * spatial_fusion + \
                 gate_weights[:, 2:3] * channel_fusion
        
        output = self.fusion(output)
        
        output = output + self.residual(x1)
        
        return output


class ImprovedTripleAttentionFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        self.weight1 = nn.Parameter(torch.tensor(0.33))
        self.weight2 = nn.Parameter(torch.tensor(0.33))
        self.weight3 = nn.Parameter(torch.tensor(0.34))
        
        self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 3, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x1, x2, x3):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        w3 = torch.sigmoid(self.weight3)
        total = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total, w2/total, w3/total
        weighted_fusion = w1 * x1 + w2 * x2 + w3 * x3
        
        concat = torch.cat([x1, x2, x3], dim=1)
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        c1 = channel_fusion[:, :x1.shape[1], :, :]
        c2 = channel_fusion[:, x1.shape[1]:x1.shape[1]*2, :, :]
        c3 = channel_fusion[:, x1.shape[1]*2:, :, :]
        channel_fusion = (c1 + c2 + c3) / 3
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x1 + \
                         spatial_weights[:, 1:2] * x2 + \
                         spatial_weights[:, 2:3] * x3
        
        alpha = torch.sigmoid(self.alpha)
        output = weighted_fusion + alpha * channel_fusion + (1 - alpha) * spatial_fusion
        
        output = self.final_conv(output)
        
        output = output + self.residual_conv(x1)
        
        return output



class HierarchicalTripleFusion(nn.Module):
    def __init__(self, in_channels=256, supplement_strength=0.2):
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
        
        self.global_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.supplement = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.supplement_strength = nn.Parameter(torch.tensor(supplement_strength))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_img, x_cb, x_sl):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1/total, w2/total
        weighted_fusion = w1 * x_img + w2 * x_cb
        
        concat = torch.cat([x_img, x_cb], dim=1)
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        channel_fusion = channel_fusion[:, :x_img.shape[1], :, :] + \
                         channel_fusion[:, x_img.shape[1]:, :, :]
        channel_fusion = channel_fusion / 2
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x_img + \
                         spatial_weights[:, 1:2] * x_cb
        
        alpha = torch.sigmoid(self.alpha)
        main_feat = weighted_fusion + alpha * channel_fusion + (1 - alpha) * spatial_fusion
        main_feat = self.global_fusion(main_feat)
        main_feat = main_feat + self.residual_conv(x_img)
        
        concat_supp = torch.cat([main_feat, x_sl], dim=1)
        supplement = self.supplement(concat_supp)
        
        strength = torch.sigmoid(self.supplement_strength)
        output = main_feat + strength * supplement
        
        output = self.final_conv(output)
        
        return output



class GatedTripleFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        self.weight1 = nn.Parameter(torch.tensor(0.33))
        self.weight2 = nn.Parameter(torch.tensor(0.33))
        self.weight3 = nn.Parameter(torch.tensor(0.34))
        
        self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 3, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.sl_encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x1, x2, x3):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        w3 = torch.sigmoid(self.weight3)
        total = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total, w2/total, w3/total
        weighted_fusion = w1 * x1 + w2 * x2 + w3 * x3
        
        concat = torch.cat([x1, x2, x3], dim=1)
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        c1 = channel_fusion[:, :x1.shape[1], :, :]
        c2 = channel_fusion[:, x1.shape[1]:x1.shape[1]*2, :, :]
        c3 = channel_fusion[:, x1.shape[1]*2:, :, :]
        channel_fusion = (c1 + c2 + c3) / 3
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x1 + \
                         spatial_weights[:, 1:2] * x2 + \
                         spatial_weights[:, 2:3] * x3
        
        alpha = torch.sigmoid(self.alpha)
        base_output = weighted_fusion + alpha * channel_fusion + (1 - alpha) * spatial_fusion
        
        gate_input = torch.cat([base_output, x3], dim=1)
        gate_weight = self.gate(gate_input)
        
        sl_feat = self.sl_encoder(x3)
        output = base_output + gate_weight * sl_feat
        
        output = self.final_conv(output)
        output = output + self.residual_conv(x1)
        
        return output



class ResidualTripleFusion(nn.Module):
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
        
        self.main_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2, x3):
        w1 = torch.sigmoid(self.weight1)
        w2 = torch.sigmoid(self.weight2)
        total = w1 + w2 + 1e-8
        w1, w2 = w1/total, w2/total
        weighted_fusion = w1 * x1 + w2 * x2
        
        concat = torch.cat([x1, x2], dim=1)
        
        channel_weights = self.channel_attn(concat)
        channel_fusion = concat * channel_weights
        channel_fusion = channel_fusion[:, :x1.shape[1], :, :] + \
                         channel_fusion[:, x1.shape[1]:, :, :]
        channel_fusion = channel_fusion / 2
        
        spatial_weights = self.spatial_attn(concat)
        spatial_fusion = spatial_weights[:, 0:1] * x1 + \
                         spatial_weights[:, 1:2] * x2
        
        alpha = torch.sigmoid(self.alpha)
        main_feat = weighted_fusion + alpha * channel_fusion + (1 - alpha) * spatial_fusion
        main_feat = self.main_fusion(main_feat)
        main_feat = main_feat + self.residual_conv(x1)
        
        residual_mask = self.residual_branch(x3)
        scale = torch.sigmoid(self.residual_scale)
        
        output = main_feat + scale * residual_mask * main_feat
        output = self.final_conv(output)
        
        return output



def get_triple_fusion_module(fusion_type, in_channels, use_improved=True, supplement_strength=0.2):
    if fusion_type == 'direct':
        if use_improved:
            return ImprovedTripleAttentionFusion(in_channels)
        else:
            return TripleAttentionFusion(in_channels)
    elif fusion_type == 'hierarchical':
        return HierarchicalTripleFusion(in_channels, supplement_strength)
    elif fusion_type == 'gated':
        return GatedTripleFusion(in_channels)
    elif fusion_type == 'residual':
        return ResidualTripleFusion(in_channels)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")