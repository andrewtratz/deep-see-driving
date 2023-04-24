from blocks import ResidualBlock, ResNet
import torch
import torch.nn as nn

##################
# backbones.py
#
# Define the ResNet backbone for the neural network, specifying number of blocks to use
##################

# Define a ResNet-like neural network model composed of residual blocks inside a modified ResNet architecture
class ResNetLike(nn.Module):
    def __init__(self):
        super(ResNetLike, self).__init__()

        # Six channel ResNet
        self.model = ResNet(6, ResidualBlock, [3, 2, 1, 1])

    def forward(self, x):
        x = self.model(x)
        return x

