
import os
from typing import Any
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision.utils import make_grid
from PIL import Image
from torch import optim


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
        self.fc1 = nn.Linear(self.num_channels * 4 * 8 * 8,
                             self.num_channels * 4)  # the output of the convolutional layers
        self.fc2 = nn.Linear(self.num_channels * 4, 6)

    def forward(self, x) -> Any:
        """At the start we do
        with an image with 3 channels and 64 x 64 pixels"""

        # convolutional layer 1
        x = self.conv1(x)  # num channels x 64 x64
        x = F.relu(F.max_pool2d(x, 2))  # parameter two say, to divide by 2 => num channels x 32 x 32

        # convolutional layer 2
        x = self.conv2(x)  # num channels*2 x 32 x32
        x = F.relu(F.max_pool2d(x, 2))  # dived by 2 => channels *2 16 x 16

        # convolutional 3
        x = self.conv3(x)  # num channels*4  x 16 x 16
        x = F.relu(F.max_pool2d(x, 2))  # num chanells x 8 x 8

        # flatten
        # -1 say we gonna do the flatten
        x = x.view(-1, self.num_channels * 4 * 8 * 8)

        # fully connected

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


# mounting dataset
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
    signs = SIGNSDataset(base_dir="datasets",
                         split="train",
                         transform=transforms.ToTensor())
    print(signs[0][0])


class RunningMetric:
    def __init__(self):
        self.S = 0
        self.N = 0

    def update(self, val, size):
        self.S += val
        self.N += size

    def __call__(self):
        return self.S / float(self.N)


# def main():
#     signs = SIGNSDataset(base_dir="datasets", 
#                         split="train", 
#                         transform=transforms.ToTensor())

#     # deliver in batches to the neuron network
#     dataloader = DataLoader(signs, batch_size=32)

#     network = Net(32)
#     loss_function = nn.NLLLoss()
#     optimizer = optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)


#     epochs = 100
#     for epoch in range(epochs):
#         print(f"Epoch => {epoch}/{epochs}")

#         running_loss = RunningMetric() # error ice in the network
#         running_acc = RunningMetric() # ice of the accuracy

#         for inputs, targets in dataloader:
#             # reload gradients to zero
#             # because in the last batch
#             # the gradients were modified
#             # insied the optimizer, so in the new batch
#             # there's to carry to zero
#             optimizer.zero_grad() 

#             outputs = network(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = loss_function(outputs, targets)
#             loss.backward() #magias: gradientes calculados automaticamente
#             optimizer.step() #magia2: actualiza las perillas o los parametros

#             batch_size = inputs.size()[0]
#             running_loss.update(loss.item()*batch_size,
#                        batch_size)
#             running_acc.update(torch.sum(preds == targets).float(),
#                        batch_size)

#         print("Loss: {:.4f} Acc: {:.4f}".format(running_loss(), running_acc()))


if __name__ == "__main__":
    main()
