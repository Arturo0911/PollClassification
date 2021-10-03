from metrics.metrics import RunningMetrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image
from torchvision import transforms
import os
from torch import optim
import torch


class ImageRegression(nn.Module):
    """linear regression for images"""

    def __init__(self, num_channels: int):
        super(ImageRegression, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)

        self.full_conn1 = nn.Linear(self.num_channels * 4 * 8 * 8, self.num_channels * 4)
        self.full_conn2 = nn.Linear(self.num_channels * 4, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, 2))

        # making flatten
        x = x.view(-1, self.num_channels * 4 * 8 * 8)

        # fully connect
        x = self.full_conn1(x)
        x = F.relu(x)

        x = self.full_conn2(x)

        x = F.log_softmax(x, dim=1)

        return x


class ImageDataset(Dataset):
    def __init__(self, basedir: str, split: str = "train", transform=None):
        path = os.path.join(basedir, f"{split}_signs")
        files = os.listdir(path)

        self.filenames = [os.path.join(path, file) for file in files if file.endswith(".jpg")]

        self.targets = [int(f[0]) for f in files]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image = Image.open(self.filenames[item])
        if self.transform:
            image = self.transform(image)
        return image, self.targets[item]


def main():

    trainset = ImageDataset(basedir="datasets", split="train", transform = transforms.ToTensor())
    dataloader = DataLoader(trainset, batch_size=32)
    net = ImageRegression(32)
    loss_n = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(100):
        print(f"Epoch {epoch+1}/100")

        running_loss = RunningMetrics()
        running_acc = RunningMetrics()

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)

            _, preds = torch.max(outputs, 1)
            loss = loss_n(outputs, targets)

            loss.backward()
            optimizer.step()


            batch_size = inputs.size()[0]
            running_loss.update(loss.item() * batch_size, batch_size)
            running_acc.update(torch.sum(preds == targets).float(), batch_size)

        print("Loss: {:.4f} Acc: {:.4f}".format(running_loss(), running_acc()))


if __name__ == "__main__":
    main()
