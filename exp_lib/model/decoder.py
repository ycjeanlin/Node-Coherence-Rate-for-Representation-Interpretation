import torch
import torch.nn as nn
from torch_geometric.nn import  Linear

from exp_lib.utils.exp_tools import set_random_seed

class LinkDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class MLPLinkDecoder(nn.Module):
    def __init__(self, input_dim, out_dim) -> None:
        set_random_seed(42)
        super(MLPLinkDecoder, self).__init__()
        self.decoder = Linear(input_dim, out_dim)
        self.bce_loss = nn.BCELoss()

    def forward(self, z, edge_index, sigmoid=True):
        h = self.decoder(z)
        value = (h[edge_index[0]] * h[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers=1, p=0.0) -> None:
        set_random_seed(42)
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers>1:
            self.layers.append(Linear(input_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.layers.append(Linear(hidden_dim, hidden_dim))
            self.layers.append(Linear(hidden_dim, out_dim))
        else:
            self.layers.append(Linear(input_dim, out_dim))

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        h = x
        for i in range(len(self.layers)-1):
            self.dropout(h)
            h = self.layers[i](h)
            h = self.act(h)
            
        out = self.layers[-1](h)
        return out

    def predict(self, x):
        out = self.forward(x)
        y_score = self.softmax(out)
        return y_score

class EnsembleMLP(nn.Module):
    def __init__(self, input_dims, out_dim, hidden_dim, p=0.0) -> None:
        super(EnsembleMLP, self).__init__()
        # self.inputs = nn.ModuleList()
        # for input_dim in input_dims:
        #     self.inputs.append(Linear(input_dim, hidden_dim))
        # self.output = Linear(hidden_dim*len(input_dims), out_dim)
        input_dim = sum(input_dims)
        self.output = Linear(input_dim, out_dim)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p)

    def reset_parameters(self):
        # for layer in self.inputs:
        #     layer.reset_parameters()
        self.output.reset_parameters()

    def forward(self, features, batch_idxs):
        hs = []
        for i in range(len(features)):
            # h = self.dropout(features[i][batch_idxs])
            # h = self.inputs[i](h)
            # h = self.act(h)
            # hs.append(h)
            hs.append(features[i][batch_idxs])
        h = torch.cat(hs, dim=-1)
        h = self.dropout(h)
        out = self.output(h)
        return out

    def predict(self, features, all_idxs):
        out = self.forward(features, all_idxs)
        y_score = self.softmax(out)
        return y_score


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes, p=0.0):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()
        for m in self.modules():
            self.weights_init(m)
        self.dropout = nn.Dropout(p)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.dropout(x)
        ret = self.fc(x)
        return ret

    def predict(self, x):
        out = self.forward(x)
        y_score = self.softmax(out)
        return y_score
        

class Predictor(nn.Module):
    def __init__(self) -> None:
        set_random_seed(42)
        super(Predictor, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return x

    def predict(self, x):
        out = self.forward(x)
        y_score = self.softmax(out)
        return y_score


