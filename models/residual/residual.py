"""ResidualBlock class"""

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Takes an input with in_channels, applies some blocks
    of convolutional layers to reduce it to out_channels and
    sum it up to the original input. If their sizes mismatch,
    then the input goes into an identity. We can abstract this
    process and create an interface that can be extended."""

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
