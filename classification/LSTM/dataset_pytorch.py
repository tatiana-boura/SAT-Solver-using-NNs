import torch
from torch.utils.data import Dataset

import pandas as pd


class SAT3Dataset(Dataset):
    def __init__(self, filename, sequence_length=5):
        self.filename = filename
        self.df = pd.read_csv(self.filename)
        self.sequence_length = sequence_length
        self.y = torch.tensor(self.df["label"])
        self.X = torch.tensor(self.df[self.df.columns[1:]].values)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
