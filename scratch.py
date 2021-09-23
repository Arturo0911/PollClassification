

import torch
from torch import nn
from torchvision import datasets
from torch.nn import functional as F


def main():
    x = torch.randn(3, 5)
    y = torch.ones_like(x)

    cifar = datasets.CIFAR10(".datasets", download=True)
    data = torch.Tensor(cifar.data)
    print(data.size())


    class Net(nn.Module):
        def __init__(self, num_channels: int):
            super(Net, self).__init__()
            self.num_channels = num_channels

            self.conv1 = 
            self.conv2 = 
            self.conv3 = 

if __name__ == "__main__":
    main()