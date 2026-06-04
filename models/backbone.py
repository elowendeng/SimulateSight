# models/backbone.py

import os
import sys
import torch
import torch.nn as nn


# Get the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_path = os.path.join(PROJECT_ROOT, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)


class Res2Net50Backbone(nn.Module):
    def __init__(self, pretrained=True, pretrained_path=None):
        super().__init__()

        # The default path for pre-trained weights
        if pretrained_path is None:
            pretrained_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'res2net50_v1b_26w_4s-3cf99910.pth')

        try:
            from Res2Net_v1b import res2net50_v1b_26w_4s
        except ImportError:
            print("Error: Please ensure that the 'Res2Net_v1b.py' file exists in lib/ directory")
            raise

        self.res2net = res2net50_v1b_26w_4s(pretrained=False)

        if pretrained and os.path.exists(pretrained_path):
            print(f"Loading Res2Net-50 pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            self.res2net.load_state_dict(new_state_dict, strict=False)
        elif pretrained:
            print(f"Warning: Pretrained weights not found at {pretrained_path}, using random initialization")

        self.layer0 = nn.Sequential(
            self.res2net.conv1,
            self.res2net.bn1,
            self.res2net.relu,
            self.res2net.maxpool
        )
        self.layer1 = self.res2net.layer1
        self.layer2 = self.res2net.layer2
        self.layer3 = self.res2net.layer3
        self.layer4 = self.res2net.layer4

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4