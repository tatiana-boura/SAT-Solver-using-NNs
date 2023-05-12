import torch
from torch.nn import Linear, LSTM, Dropout, Sequential, ReLU, ModuleList

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
class ShallowLSTM(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super().__init__()

        self.num_layers = model_params["model_num_layers"]
        self.hidden_units = model_params["model_hidden_units"]
        self.n_deep_layers = model_params["model_deep_layers"]
        dropout = model_params["model_dropout"]
        self.hidden = None
        self.sequence_len = 2

        # LSTM Layer
        self.lstm = LSTM(input_size=feature_size, hidden_size=self.hidden_units, dropout=dropout,
                         num_layers=self.num_layers, batch_first=True)

        # first dense after lstm
        self.fc1 = Linear(self.hidden_units * self.sequence_len, self.hidden_units)
        # Dropout layer
        self.dropout = Dropout(p=dropout)

        # Create fully connected layers (n_hidden x n_deep_layers)
        # dnn_layers = []
        dnn_layers = ModuleList([])
        for i in range(self.n_deep_layers):
            # last layer (n_hidden x 1)
            if i == self.n_deep_layers - 1:
                dnn_layers.append(ReLU())
                dnn_layers.append(Linear(self.hidden_units, 1))
            # all other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(ReLU())
                dnn_layers.append(Linear(self.hidden_units, self.hidden_units))
                if self.dropout:
                    dr = Dropout(p=dropout)
                    dnn_layers.append(dr)
        # compile DNN layers
        self.dnn = Sequential(*dnn_layers)

    def forward(self, x):

        # Initialize hidden state
        hidden_state = torch.zeros(self.num_layers, x.shape[0], self.hidden_units)
        cell_state = torch.zeros(self.num_layers, x.shape[0], self.hidden_units)

        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)

        self.hidden = (hidden_state, cell_state)

        # Forward Pass
        x, h = self.lstm(x, self.hidden)  # LSTM
        x = self.dropout(x.contiguous().view(x.shape[0], -1))  # Flatten lstm out
        x = self.fc1(x)  # First Dense

        return self.dnn(x)  # Pass forward through fully connected DNN.


'''
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
