

import torch
from torch import nn



def main():
    x = torch.ones(1)
    linear_model = nn.Linear(1, 1)
    print(linear_model(x))


if __name__ == "__main__":
    main()