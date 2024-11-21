import torch
import numpy as np
import torch.nn as nn

VGGs = {
    "VGG11": [1, 1, 2, 2, 2], # 1x64, 1x128, 2x256, 2x512, 2x512
    "VGG13": [2, 2, 2, 2, 2], # 2x64, 2x128, 2x256, 2x512, 2x512
    "VGG16": [2, 2, 3, 3, 3], # 2x64, 2x128, 3x256, 3x512, 3x512
    "VGG19": [2, 2, 4, 4, 4]  # 2x64, 2x128, 4x256, 4x512, 4x512
}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, no_layers):
        super().__init__()
        self.relu = nn.ReLU()
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                  self.relu]
        for _ in no_layers:
            layers.append([nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)])
            layers.append(self.relu)

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)
    

block = Block(3, 64, )