# models/decoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .single_attention import get_attention_module
from .dual_fusion import get_dual_fusion_module
from .triple_fusion import get_triple_fusion_module


class AttentionSingleDecoder(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], attn_type='dual', use_improved=True):
        super().__init__()
        c1, c2, c3, c4 = in_channels

        self.attn1 = get_attention_module(attn_type, c1, use_improved)
        self.attn2 = get_attention_module(attn_type, c2, use_improved)
        self.attn3 = get_attention_module(attn_type, c3, use_improved)
        self.attn4 = get_attention_module(attn_type, c4, use_improved)

        self.side1 = nn.Conv2d(c1, 32, 3, padding=1)
        self.side2 = nn.Conv2d(c2, 32, 3, padding=1)
        self.side3 = nn.Conv2d(c3, 32, 3, padding=1)
        self.side4 = nn.Conv2d(c4, 32, 3, padding=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self.attn_type = attn_type

    def forward(self, feats, target_size=None):
        x1, x2, x3, x4 = feats

        x1 = self.attn1(x1)
        x2 = self.attn2(x2)
        x3 = self.attn3(x3)
        x4 = self.attn4(x4)

        mid_size = x1.shape[2:]

        s1 = self.side1(x1)
        s2 = F.interpolate(self.side2(x2), size=mid_size, mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(x3), size=mid_size, mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(x4), size=mid_size, mode='bilinear', align_corners=False)

        fuse = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.fuse(fuse)

        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out


class AttentionDualDecoder(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], attn_type='spatial', use_improved=True):
        super().__init__()

        c1, c2, c3, c4 = in_channels

        self.fusion1 = get_dual_fusion_module(attn_type, c1, use_improved)
        self.fusion2 = get_dual_fusion_module(attn_type, c2, use_improved)
        self.fusion3 = get_dual_fusion_module(attn_type, c3, use_improved)
        self.fusion4 = get_dual_fusion_module(attn_type, c4, use_improved)

        self.reduce1 = nn.Conv2d(c1, 32, 1)
        self.reduce2 = nn.Conv2d(c2, 32, 1)
        self.reduce3 = nn.Conv2d(c3, 32, 1)
        self.reduce4 = nn.Conv2d(c4, 32, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, feats1, feats2, target_size=None):
        f1 = self.fusion1(feats1[0], feats2[0])
        f2 = self.fusion2(feats1[1], feats2[1])
        f3 = self.fusion3(feats1[2], feats2[2])
        f4 = self.fusion4(feats1[3], feats2[3])

        mid_size = f1.shape[2:]

        s1 = self.reduce1(f1)
        s2 = F.interpolate(self.reduce2(f2), size=mid_size, mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.reduce3(f3), size=mid_size, mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.reduce4(f4), size=mid_size, mode='bilinear', align_corners=False)

        fuse = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.fuse(fuse)

        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out


class AttentionTripleDecoder(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], fusion_type='direct', 
                 use_improved=True, supplement_strength=0.2):
        super().__init__()
        
        c1, c2, c3, c4 = in_channels

        self.fusion1 = get_triple_fusion_module(fusion_type, c1, use_improved, supplement_strength)
        self.fusion2 = get_triple_fusion_module(fusion_type, c2, use_improved, supplement_strength)
        self.fusion3 = get_triple_fusion_module(fusion_type, c3, use_improved, supplement_strength)
        self.fusion4 = get_triple_fusion_module(fusion_type, c4, use_improved, supplement_strength)
        
        self.reduce1 = nn.Conv2d(c1, 32, 1)
        self.reduce2 = nn.Conv2d(c2, 32, 1)
        self.reduce3 = nn.Conv2d(c3, 32, 1)
        self.reduce4 = nn.Conv2d(c4, 32, 1)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        self.skip_connection = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feats1, feats2, feats3, target_size=None):
        f1 = self.fusion1(feats1[0], feats2[0], feats3[0])
        f2 = self.fusion2(feats1[1], feats2[1], feats3[1])
        f3 = self.fusion3(feats1[2], feats2[2], feats3[2])
        f4 = self.fusion4(feats1[3], feats2[3], feats3[3])
        
        mid_size = f1.shape[2:]
        
        s1 = self.reduce1(f1)
        s2 = F.interpolate(self.reduce2(f2), size=mid_size, mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.reduce3(f3), size=mid_size, mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.reduce4(f4), size=mid_size, mode='bilinear', align_corners=False)
        
        fuse = torch.cat([s1, s2, s3, s4], dim=1)
        
        out = self.fuse(fuse)
        skip = self.skip_connection(fuse)
        out = out + 0.3 * skip
        
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out


class AuxiliaryDecoder(nn.Module):
    """Lightweight auxiliary decoder"""

    def __init__(self, in_channels=[256, 512, 1024, 2048]):
        super().__init__()

        c1, c2, c3, c4 = in_channels

        self.reduce1 = nn.Conv2d(c1, 32, 1)
        self.reduce2 = nn.Conv2d(c2, 32, 1)
        self.reduce3 = nn.Conv2d(c3, 32, 1)
        self.reduce4 = nn.Conv2d(c4, 32, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, feats, target_size=None):
        f1, f2, f3, f4 = feats
        mid_size = f1.shape[2:]

        s1 = self.reduce1(f1)
        s2 = F.interpolate(self.reduce2(f2), size=mid_size, mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.reduce3(f3), size=mid_size, mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.reduce4(f4), size=mid_size, mode='bilinear', align_corners=False)

        fuse = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.fuse(fuse)

        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out