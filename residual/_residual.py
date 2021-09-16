"""ResidualBlock class"""

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
