"""Growth bacterian prediction"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader
)
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "datasets/data_regression_set.csv"
NUM_EPOCHS = 150
BATCH_SIZE = 64


class BacterianDataset(Dataset):
    def __init__(self, features, targets, transform=None):
        self.features = features
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LinearRegression(nn.Module):
    def __init__(self, num_features: int):
        super(LinearRegression, self).__init__()
        self.layer1 = nn.Linear(num_features, 12)
        self.layer2 = nn.Linear(12, 24)
        self.layer3 = nn.Linear(24, 12)
        self.latyer_out = nn.Linear(12, 1)
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.latyer_out(x)
        return x

    def predict(self, test_inputs):
        x = self.relu(self.layer1(test_inputs))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.latyer_out(x)
        return x

def main():
    # defining a dataset to make split in the features
    # and the targets in the values
    def split_dataset(path_dir) -> list:
        dataset = pd.read_csv(path_dir)
        features = dataset.drop(["quality_product", "lactobacillus_final_cfu_ml", 
                       "streptococcus_final_cfu_ml", 
                       "lactobacillus_initial_strain_cfu_ml"], axis=1)
        targets = dataset["streptococcus_final_cfu_ml"]
        return train_test_split(features, targets, test_size=0.3, shuffle=True, random_state=42)
    
    X_train, X_test, y_train, y_test = split_dataset(DATA_PATH)
    NUM_FEATURES = len(X_train.columns)
    print(X_train.columns)
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    train_dataset = BacterianDataset(torch.from_numpy(X_train).float(), 
                                    torch.from_numpy(y_train).float())

    test_dataset = BacterianDataset(torch.from_numpy(X_test).float(), 
                                    torch.from_numpy(y_test).float())

    
    data_loader = DataLoader(dataset=train_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    # print(data_loader)
    re_model = LinearRegression(NUM_FEATURES)

    print("\n\nTRAININ GONNA BE START\n\n")

    



if __name__ == "__main__":
    main()