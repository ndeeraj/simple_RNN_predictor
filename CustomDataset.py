import os
import pathlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas import read_csv
import torch

import numpy as np
from torch.utils.data import Dataset


class CustomDataSt(Dataset):
    normalizer = None

    def __init__(self, data_frame, normalizer=None, window=5, plot_data=False):
        self.df = data_frame
        self.window = window
        self.normalizer = normalizer
        self.unnorm_val = []
        self.norm_val = []
        self.sequences = {}
        self.create_inout_sequences()
        self.data_stat()
        if plot_data:
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
        window = self.window
        self.unnorm_val = np.array(self.df["Receipt_Count"]).astype('float64')
        if self.normalizer is None:
            self.normalizer = preprocessing.StandardScaler().fit(
                self.unnorm_val.reshape(len(self.unnorm_val), 1))
        self.norm_val = self.normalizer.transform(self.unnorm_val.reshape(len(self.unnorm_val), 1))
        values = np.copy(self.norm_val)

        size = len(values)
        for i in range(size - window):
            sequence = {"seq": values[i:i + window], "out": values[i + window:i + window + 1]}
            self.sequences[i] = sequence
        return

    def data_stat(self):
        min_val = min(self.unnorm_val)
        max_val = max(self.unnorm_val)
        num_samples = len(self.unnorm_val)

        print("\nNumber of samples: " + str(num_samples))
        print("Max receipt count: " + str(max_val))
        print("Min receipt count: " + str(min_val))

    def plot_data(self):
        x = np.arange(0, len(self.unnorm_val))
        plt.scatter(x, self.unnorm_val)
        plt.ylabel("Receipt count")
        plt.xlabel("Samples")
        plt.title("Receipt counts in the dataset.")
        plt.show()


def split_data_to_dataframes(path_to_data_file: str, split_factor: float = 0.1) -> list:
    """
    Splits the data in the provided data file into test, validation and training data.

    :param path_to_data_file: (str) absolute path to the data file
    :param split_factor: (float) between 0-0.33; this percentage will be used on the data length
                        to create test, validation and train data frames.
                        The first split will be test data, the second will be validation and the
                        rest would be training data.
    :return: (list) of dataframes ordered as train, validation, train data frames.
    """
    if split_factor > 0.33 or split_factor < 0:
        raise RuntimeError("split factor is invalid.")

    data_df = read_csv(path_to_data_file)
    split_index_test = round(split_factor * len(data_df))

    try:
        test_df = data_df.iloc[:split_index_test]
        split_index_val = 2 * split_index_test
        val_df = data_df.iloc[split_index_test:split_index_val]
        train_df = data_df.iloc[split_index_val:]

        return [train_df, val_df, test_df]
    except IndexError as exp:
        raise IndexError(f'''Cannot split by {split_factor} because it results in index out of \
        bounds errors on the source data.\n\nmore details on the error:\n{str(exp)}''')


if __name__ == "__main__":
    project_root = pathlib.Path().resolve()
    dataDir = 'data'
    dataFile = 'data_daily.csv'

    path_to_file = os.path.join(project_root, dataDir, dataFile)

    (train_df, val_df, test_df) = split_data_to_dataframes(path_to_file)

    train_dtst = CustomDataSt(train_df)
    val_dtst = CustomDataSt(val_df)
    test_dtst = CustomDataSt(test_df)

