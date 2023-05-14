import torch
from torch.nn import Linear, LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ShallowLSTM(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super().__init__()
        self.hidden_units = model_params["model_hidden_units"]
        self.num_layers = model_params["model_num_layers"]
        self.dropout = model_params["model_dropout"]

        self.lstm = LSTM(input_size=feature_size, hidden_size=self.hidden_units,
                         num_layers=self.num_layers, dropout=self.dropout, batch_first=True)

        self.linear = Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        h0 = h0.to(device)
        c0 = c0.to(device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
