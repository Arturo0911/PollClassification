
import numpy as np
import pandas as pd
import glob
import itertools
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader
)

from utils.utils import ReplayBuffer

from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, base_dir, transform=None, split="train"):
        self.transform = transforms.Compose(transform)
        self.file_A = glob.glob(os.path.join(base_dir, '{}/A/*.*'.format(split)))



def main():
    pass


if __name__ == "__main__":
    main()
