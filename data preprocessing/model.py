import torch
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

torch.manual_seed(15)

class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        embedding_size = 64
        n_heads = 3
        self.n_layers = 4
        dropout_rate = 0.2
        self.top_k_every_n = 1
        dense_neurons = 256
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # transformation layer
        self.conv1 = TransformerConv(feature_size, embedding_size, heads=n_heads, dropout=dropout_rate, edge_dim=edge_dim, beta=True)
        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        # we add Batch Normalization after the activation function of the output layer
        self.bn1 = BatchNorm1d(embedding_size)

        # other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size, embedding_size, heads=n_heads, dropout=dropout_rate, edge_dim=edge_dim, beta=True))
            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

        # linear layers
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # holds the intermediate graph representations
        global_representation = []
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        x = sum(global_representation)

        # output block
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x