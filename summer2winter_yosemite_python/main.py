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

from models.generator.generator import Generator
from models.discriminator.discriminator import Discriminator


class ImageDataset(Dataset):
    def __init__(self, base_dir, transform=None, split="train"):
        self.transform = transforms.Compose(transform)

        self.file_A = sorted(glob.glob(os.path.join(base_dir,
                                                    '{}/A/*.*'.format(split))))

        self.file_B = sorted(glob.glob(os.path.join(base_dir,
                                                    '{}/B/*.*'.format(split))))

    def __len__(self):
        return max(len(self.file_A), len(self.file_B))

    def __getitem__(self, idx):
        image_A = self.transform(Image.open(self.file_A[idx]))
        image_B = self.transform(Image.open(self.file_B[random.randint(0, len(self.file_B) - 1)]))
        return {"A": image_A, "B": image_B}


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != 1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != 1:
        torch.nn.init.normal((m.weight.data, 1.0, 0.02))
        torch.nn.init.constant(m.bias, 0.0)


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

    optimizer_G = torch.optim.Adam(itertools.chain(
        netG_A2B.parameters(), netG_B2A.parameters()),
        lr=learning_rate, betas=(0.5, 0.999))

    optimizer_D_A = torch.optim.Adam(netD_A.parameters(),
                                     lr=learning_rate,
                                     betas=(0.5, 0.999))

    optimizer_D_B = torch.optim.Adam(netD_B.parameters(),
                                     lr=learning_rate,
                                     betas=(0.5, 0.999))

    # schedulers (updating learning rate by dynamic while the training)
    # LambdaLR how much time the learning rate will be updating.

    class LambdaLR:
        def __init__(self, n_epochs, offset, decay_start_epoch):
            assert ((n_epochs - decay_start_epoch) > 0)
            self.n_epochs = n_epochs
            self.offset = offset
            self.decay_start_epoch = decay_start_epoch

        def step(self, epoch):
            return 1 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(n_epochs,
                                                                          epoch,
                                                                          decay_epoch).step())
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                       lr_lambda=LambdaLR(n_epochs,
                                                                          epoch,
                                                                          decay_epoch).step())
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                       lr_lambda=LambdaLR(n_epochs,
                                                                          epoch,
                                                                          decay_epoch).step())
    # inputs and targets
    Tensor = torch.FloatTensor
    target_real = Tensor(batch_size).fill_(1.0)
    target_fake = Tensor(batch_size).fill_(0.0)

    # to store fake images
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Data loader
    transform = [transforms.Resize(int(size * 1.12), Image.BICUBIC),
                 transforms.RandomCrop(size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(ImageDataset(base_dir, transform=transform),
                            batch_size=batch_size, shuffle=True, num_workers=n_cpu, drop_last=True)

    def Gen_GAN_loss(G, D, real, loss, target_real):
        fake = G(real)
        pred_fake = D(fake)
        L = loss(pred_fake, target_real)
        return L, fake


if __name__ == "__main__":
    main()
