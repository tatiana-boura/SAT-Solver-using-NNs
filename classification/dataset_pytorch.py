from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os


class SAT3Dataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        self.filename = filename
        super(SAT3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return "no_implemented_stuff.pt"

    def download(self):
        # there won't be any need to download the data
        pass

    def process(self):
        # open the dataframe
        store = pd.HDFStore(self.filename)
        self.data = store['df']

        for index, cnf in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # get node features (here we actually don't have many)
            node_feats = torch.tensor(cnf["variablesSymb"], dtype=torch.float)
            # get adjacency info
            edge_index = torch.tensor(cnf["edges"], dtype=torch.long)
            num_edges = edge_index.size(dim=1)
            # get edge features |  view is used in order to get the correct dimensions as specified by COO format
            edge_feats = torch.tensor(cnf["edgeAttr"], dtype=torch.float).view(num_edges, -1)
            # get labels info
            label = torch.tensor(np.asarray(cnf["label"]), dtype=torch.int64)
            # now, create data object
            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=label)
            # save the data
            torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
