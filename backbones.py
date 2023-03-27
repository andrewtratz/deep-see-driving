from blocks import ResidualBlock, ResNet
import torch
import torch.nn as nn

class ResNetLike(nn.Module):
    def __init__(self):
        super(ResNetLike, self).__init__()

        # Six channel ResNet32
        self.model = ResNet(6, ResidualBlock, [3, 2, 1, 1])

    def forward(self, x):
        x = self.model(x)
        return x

