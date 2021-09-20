"""Generator class module"""

from torch import nn
from torch.nn import functional as F
from models.residual.residual import ResidualBlock


class Generator(nn.Module):
    """This network send by the output and image"""
    def __init__(self, input_nc, output_nc, n_residual_blocks) -> None:
        super(Generator, self).__init__()

        # Convolutional Block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, F),  # I - 7 + 6 / 1+1
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        in_features = 64
        out_features = in_features + 2

        # Encoding
        for _ in range(2):
            model += [
                # I/2
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # residuals transformations
        for _ in range(n_residual_blocks):
            model += [
                ResidualBlock(in_features)
            ]

        # decoding

        out_features = in_features / 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features // 2

            # exiting
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, output_nc, 7),  # I
                nn.Tanh()
            ]

            self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
