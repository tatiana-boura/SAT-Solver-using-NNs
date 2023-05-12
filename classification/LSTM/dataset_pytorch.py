import torch
from torch.utils.data import Dataset

import pandas as pd


class SAT3Dataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(self.filename)
        self.sequence_length = 2
        self.y = torch.tensor(self.df["label"])
        self.X = torch.tensor(self.df[self.df.columns[1:]].values)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i+1 != len(self) and ((self.X[i][0]).item() == (self.X[i+1][0]).item()):
            x = self.X[i:(i+2), :]
            y = self.y[i+1]
        else:
            x1 = self.X[i - 1, :]
            x2 = self.X[i, :]
            x = torch.stack([x1, x2])
            y = self.y[i]

        return x, y