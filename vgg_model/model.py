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
        self.layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                  self.relu]
        for _ in range(no_layers):
            self.layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.layers.append(self.relu)

        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.conv_block(x)
    

class VGG(nn.Module):
    def __init__(self,
                 model_type: str,
                 in_channels: int = 3,
                 classes: int = 1000):
        super().__init__()

        configs = {
            "VGG11": [1, 1, 2, 2, 2], # 1x64, 1x128, 2x256, 2x512, 2x512
            "VGG13": [2, 2, 2, 2, 2], # 2x64, 2x128, 2x256, 2x512, 2x512
            "VGG16": [2, 2, 3, 3, 3], # 2x64, 2x128, 3x256, 3x512, 3x512
            "VGG19": [2, 2, 4, 4, 4]  # 2x64, 2x128, 4x256, 4x512, 4x512
        }

        config = configs[model_type]
        channels = [in_channels, 64, 128, 256, 512, 512]
        
        self.blocks = nn.ModuleList([])

        for i in range(len(config)):
            self.blocks.append(Block(channels[i], channels[i+1], config[i]))

        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.relu(self.fc3(x))

        return x


if __name__ == "__main__":

    vgg = VGG(model_type="VGG11")

    input = torch.rand(1, 3, 244, 244)
    output = vgg(input)
    print(output.shape)

