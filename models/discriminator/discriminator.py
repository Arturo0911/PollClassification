"""Discriminator module"""

from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    """The job of the discriminator is to look
    at an image and output whether or not it is
    a real training image or fake image from the
    generator

    >> Discriminative network, gonna send numbers
        int the output by the flatten.
    """
    def __init__(self, input_nc) -> None:
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Flatten
        model += [
            nn.Conv2d(512, 1, 4, padding=1) # I - 1
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)