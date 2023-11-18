import os
import pathlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas import read_csv
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataSt(Dataset):
    def __init__(self, data_frame, transform=None):
        self.transform = transform
        self.df = data_frame
        self.unnorm_val = []
        self.norm_val = []
        self.sequences = {}
        self.create_inout_sequences()
        self.data_stat()
        self.plot_data()

    def report_data_desc(self):
        self.data_stat()

    def __len__(self):
        return len(self.sequences.keys())

    def __getitem__(self, idx):
        features = torch.tensor(self.sequences[idx]["seq"], dtype=torch.double)
        output = torch.tensor(self.sequences[idx]["out"], dtype=torch.double)

        return features, output

    def create_inout_sequences(self):
        window = 10
        self.unnorm_val = np.array(self.df["High"]).astype('float32')
        std_scalar = preprocessing.StandardScaler().fit(self.unnorm_val.reshape(len(self.unnorm_val), 1))
        self.norm_val = std_scalar.transform(self.unnorm_val.reshape(len(self.unnorm_val), 1))
        values = np.copy(self.norm_val)
        L = len(values)
        for i in range(L-window):
            sequence = {"seq": values[i:i+window], "out": values[i+window:i+window + 1]}
            self.sequences[i] = sequence
        return

    def data_stat(self):
        min_val = min(self.unnorm_val)
        max_val = max(self.unnorm_val)
        num_samples = len(self.unnorm_val)

        print("\nNumber of samples: " + str(num_samples))
        print("Max stock value: " + str(max_val))
        print("Min stock value: " + str(min_val))

    def plot_data(self):
        x = np.arange(0, len(self.unnorm_val))
        plt.scatter(x, self.unnorm_val)
        plt.show()

if __name__ == "__main__":
    project_root = pathlib.Path().resolve()
    dataDir = 'data'
    testFile = 'test.csv'
    trainFile = 'train.csv'
    validationFile = 'validation.csv'
    path_to_file = os.path.join(project_root, dataDir)

    test_df = read_csv(os.path.join(path_to_file, testFile))
    train_df = read_csv(os.path.join(path_to_file, trainFile))
    val_df = read_csv(os.path.join(path_to_file, validationFile))

    train_dt_st = CustomDataSt(train_df)