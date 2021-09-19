"""ResidualBlock class"""

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),  # best padding
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),  # Bn for GANS
            nn.ReLU(True),
            nn.ReflectionPad2d(1),  # best for preserve the distribution
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x) + x
