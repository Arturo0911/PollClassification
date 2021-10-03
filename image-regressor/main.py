# from metrics.metrics import RunningMetrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image
from torchvision import transforms
import os




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

        self.filenames = [os.path.join(basedir, file) for file in files if file.endswith(".jpg")]

        self.target = [int(f[0]) for f in files]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image = Image.open(self.filenames[item])
        if self.transform:
            image = self.transform(image)
        return image


def main():

    trainset = ImageDataset(basedir="datasets", split="train", transform = transforms.ToTensor())
    dataloader = DataLoader(trainset, batch_size=32)

    for inputs, targets in dataloader:
        print(inputs, targets)


if __name__ == "__main__":
    main()
