# models/models.py

import torch.nn as nn
from .backbone import Res2Net50Backbone
from .decoders import AttentionSingleDecoder, AttentionDualDecoder, AttentionTripleDecoder, AuxiliaryDecoder


class AttentionSingleStreamCOD(nn.Module):
    def __init__(self, pretrained=True, attn_type='dual', input_size=352, use_improved=True):
        super().__init__()
        self.backbone = Res2Net50Backbone(pretrained)
        self.decoder = AttentionSingleDecoder(attn_type=attn_type, use_improved=use_improved)
        self.input_size = input_size

    def forward(self, x):
        feats = self.backbone(x)
        target_size = x.shape[2:]
        out = self.decoder(feats, target_size=target_size)
        return out


class AttentionDualStreamCOD(nn.Module):
    def __init__(self, pretrained=True, attn_type='spatial', input_size=352,
                 share_backbone=True, use_improved=True):
        super().__init__()
        self.share_backbone = share_backbone
        self.input_size = input_size
        self.use_improved = use_improved

        if share_backbone:
            self.backbone = Res2Net50Backbone(pretrained)
        else:
            self.backbone1 = Res2Net50Backbone(pretrained)
            self.backbone2 = Res2Net50Backbone(pretrained)

        self.decoder = AttentionDualDecoder(attn_type=attn_type, use_improved=use_improved)

    def forward(self, x1, x2):
        if self.share_backbone:
            feats1 = self.backbone(x1)
            feats2 = self.backbone(x2)
        else:
            feats1 = self.backbone1(x1)
            feats2 = self.backbone2(x2)

        target_size = x1.shape[2:]
        out = self.decoder(feats1, feats2, target_size=target_size)
        return out

    def extract_features(self, x1, x2):
        if self.share_backbone:
            feats1 = self.backbone(x1)
            feats2 = self.backbone(x2)
        else:
            feats1 = self.backbone1(x1)
            feats2 = self.backbone2(x2)
        return feats1, feats2


class AttentionDualStreamCODWithAux(nn.Module):
    """Dual-stream model with auxiliary loss"""

    def __init__(self, pretrained=True, attn_type='spatial', input_size=352,
                 share_backbone=True, use_improved=True, use_aux_loss=True):
        super().__init__()
        self.share_backbone = share_backbone
        self.input_size = input_size
        self.use_improved = use_improved
        self.use_aux_loss = use_aux_loss

        if share_backbone:
            self.backbone = Res2Net50Backbone(pretrained)
        else:
            self.backbone1 = Res2Net50Backbone(pretrained)
            self.backbone2 = Res2Net50Backbone(pretrained)

        self.decoder = AttentionDualDecoder(attn_type=attn_type, use_improved=use_improved)

        if use_aux_loss:
            self.aux_decoder_img = AuxiliaryDecoder([256, 512, 1024, 2048])
            self.aux_decoder_cb = AuxiliaryDecoder([256, 512, 1024, 2048])
            print("Using auxiliary decoders for each stream")

    def forward(self, x1, x2, return_aux=False):
        if self.share_backbone:
            feats1 = self.backbone(x1)
            feats2 = self.backbone(x2)
        else:
            feats1 = self.backbone1(x1)
            feats2 = self.backbone2(x2)

        target_size = x1.shape[2:]
        main_out = self.decoder(feats1, feats2, target_size=target_size)

        if return_aux and self.use_aux_loss:
            aux_out1 = self.aux_decoder_img(feats1, target_size=target_size)
            aux_out2 = self.aux_decoder_cb(feats2, target_size=target_size)
            return main_out, [aux_out1, aux_out2]

        return main_out

    def extract_features(self, x1, x2):
        if self.share_backbone:
            feats1 = self.backbone(x1)
            feats2 = self.backbone(x2)
        else:
            feats1 = self.backbone1(x1)
            feats2 = self.backbone2(x2)
        return feats1, feats2


class AttentionTripleStreamCOD(nn.Module):
    def __init__(self, pretrained=True, input_size=352, 
                 share_backbone=True, use_improved=True, use_aux_loss=False,
                 fusion_type='direct', supplement_strength=0.2):
        super().__init__()
        self.share_backbone = share_backbone
        self.input_size = input_size
        self.use_improved = use_improved
        self.use_aux_loss = use_aux_loss
        self.fusion_type = fusion_type
        
        if share_backbone:
            self.backbone = Res2Net50Backbone(pretrained)
        else:
            self.backbone1 = Res2Net50Backbone(pretrained)
            self.backbone2 = Res2Net50Backbone(pretrained)
            self.backbone3 = Res2Net50Backbone(pretrained)
        
        self.decoder = AttentionTripleDecoder(
            in_channels=[256, 512, 1024, 2048],
            fusion_type=fusion_type,
            use_improved=use_improved,
            supplement_strength=supplement_strength
        )
        
        if fusion_type == 'direct':
            if use_improved:
                print("Using Direct Fusion with ImprovedTripleAttentionFusion")
            else:
                print("Using Direct Fusion with TripleAttentionFusion (original with gate)")
        else:
            print(f"Using {fusion_type} fusion")
        
        if use_aux_loss:
            self.aux_decoder_img = AuxiliaryDecoder()
            self.aux_decoder_cb = AuxiliaryDecoder()
            self.aux_decoder_sl = AuxiliaryDecoder()
            print("Using auxiliary decoders for each stream")
    
    def extract_features(self, x1, x2, x3):
        if self.share_backbone:
            return self.backbone(x1), self.backbone(x2), self.backbone(x3)
        else:
            return self.backbone1(x1), self.backbone2(x2), self.backbone3(x3)
        
    def forward(self, x1, x2, x3, return_aux=False):
        if self.share_backbone:
            feats1 = self.backbone(x1)
            feats2 = self.backbone(x2)
            feats3 = self.backbone(x3)
        else:
            feats1 = self.backbone1(x1)
            feats2 = self.backbone2(x2)
            feats3 = self.backbone3(x3)
        
        target_size = x1.shape[2:]
        main_out = self.decoder(feats1, feats2, feats3, target_size=target_size)
        
        if return_aux and self.use_aux_loss:
            aux_out1 = self.aux_decoder_img(feats1, target_size=target_size)
            aux_out2 = self.aux_decoder_cb(feats2, target_size=target_size)
            aux_out3 = self.aux_decoder_sl(feats3, target_size=target_size)
            return main_out, [aux_out1, aux_out2, aux_out3]
        
        return main_out