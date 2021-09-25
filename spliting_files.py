"""Spliting csv files in directories for train, test and validation"""


import os
import pandas


PATH_DIR = "datasets/data_regression_set.csv"


def spliting_files():
    dataset = pd.read_csv(PATH_DIR)
    train_dataset = ""
    test_dataset = ""


spliting_files()