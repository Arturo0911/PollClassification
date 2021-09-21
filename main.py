
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
import os
from torchvision.transforms import transforms
import random
import torch


from generator.generator import Generator
from discriminator.discriminator import Discriminator



class ImageDataset(Dataset):
    def __init__(self, base_dir, transform=None, split="train"):
        self.transform = transforms.Compose(transform)

        self.file_A = sorted(glob.glob(os.path.join(base_dir, 
                            '{}/A/*.*'.format(split))))

        self.file_B = sorted(glob.glob(os.path.join(base_dir, 
                            '{}/B/*.*'.format(split))))

    def __len__(self):
        return max(self.file_A, len(self.file_B))
    

    def __getitem__(self, idx):
        image_A = self.transform(Image.open(self.file_A[idx]))
        image_B = self.transform(Image.open(self.file_B[random.randint(0, len(self.file_B) - 1)]))
        return {"A"; image_A, "B":image_B} 



def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") !=1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != 1:
        torch.nn.init.normal((m.weight.data, 1.0, 0.02))
        torch.nn.init.constant(m.bias, 0.0)

    # netG_A2B = Generator(input_nc, output_nc)
    # netG_B2A = Generator(input_nc, output_nc)
    # netD_A = Discriminator(input_nc)
    # netD_B = Discriminator(input_nc)


    # netG_A2B.apply(weights_init_normal) 
    # netG_B2A.apply(weights_init_normal)
    # netD_A.apply(weights_init_normal) 
    # netD_B.apply(weights_init_normal)


def main():
    epoch = 0
    n_epochs = 200
    batch_size = 4
    learning_rate = 0.0002
    size = 256
    input_nc = 3
    output_nc = 3
    decay_epoch = 100

    base_dir = "summer2winter_yosemite/"

    netG_A2B = Generator(input_nc, output_nc)
    netG_B2A = Generator(input_nc, output_nc)
    netD_A = Discriminator(input_nc)
    netD_B = Discriminator(input_nc)


    netG_A2B.apply(weights_init_normal) 
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal) 
    netD_B.apply(weights_init_normal)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()


if __name__ == "__main__":
    main()
