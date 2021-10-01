#!/usr/bin/env python

import torch
from torch import nn
from torch.utils.data import Dataset

from PIL import Image
import imageio
import os
from pprint import pprint


# class ImageClassifier(nn.Module):

#     def __init__(self, num_channels: int):
#         super(ImageClassifier, self).__init__()

#         self.conv1 = nn.Conv2d(num_channels, num_channels*2, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding=1)


#     def forward(self, x):

def main():

    image_files = [data for data in os.listdir("datasets/train_signs") if os.path.splitext(data)[-1] == ".jpg"]

    for x in image_files:
        read_image = imageio.imread(os.path.join("datasets/train_signs", x))
        print(torch.from_numpy(read_image).float().shape)
    # read_image = imageio.imread("datasets/train_signs/0_IMG_5864.jpg")
    # tensor_image = torch.from_numpy(read_image).float()
    # print(tensor_image.shape)

if __name__ == "__main__":
    main()
