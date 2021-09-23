

import os
from typing import Any
import torch
from torch import nn
from torchvision import datasets
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image



class Net(nn.Module):
    def __init__(self, num_channels: int):
        super(Net, self).__init__()
        self.num_channels = num_channels

        # this network gonna take three layers
        # convolutional
        # features extractors
        # here num channels is 3 as the input is an image
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels *2, self.num_channels * 4, 3, stride=1, padding=1)


        # fully connectted linear
        self.fc1 = nn.Linear(self.num_channels*4*8*8, self.num_channels*4) # the output of the convolutional layers
        self.fc2 = nn.Linear(self.num_channels*4, 6)
    
    def forward(self, x) -> Any:
        """At the start we do
        with an image with 3 channels and 64 x 64 pixels"""

        # convolutional layer 1
        x = self.conv1(x) # num channels x 64 x64
        x = F.relu(F.max_pool2d(x, 2)) # parameter two say, to divide by 2 => num channels x 32 x 32

        # convolutional layer 2
        x = self.conv2(x) # num channels*2 x 32 x32
        x = F.relu(F.max_pool2d(x, 2)) # dived by 2 => channels *2 16 x 16

        # convolutional 3
        x = self.conv3(x) # num channels*4  x 16 x 16
        x = F.relu(F.max_pool2d(x, 2)) # num chanells x 8 x 8

        # flatten
        # -1 say we gonna do the flatten
        x = x.view(-1, self.num_channels*4*8*8)

        # fully connected

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x



# mouting dataset
class SIGNSDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        path = os.path.join(base_dir, "{}_signs".format(split))
        
        files = os.listdir(path)
        self.filenames = [os.path.join(path, f) for f in files if f.endswith(".jpg")]
        self.targets = [int(f[0]) for f in files]
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        if self.transform:
            image = self.transform(image)
        
        return image, self.targets[idx]


def main():
    # x = torch.randn(3, 5)
    # y = torch.ones_like(x)

    # cifar = datasets.CIFAR10(".datasets", download=True)
    # data = torch.Tensor(cifar.data)
    # print(data.size())
    signs = SIGNSDataset(base_dir="datasets", split="train")
    print(len(signs))
    print(signs[0][0])


if __name__ == "__main__":
    main()