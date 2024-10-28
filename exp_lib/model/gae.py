from typing import Any
import torch
import torch.nn as nn
from torch.nn import Embedding
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling

from exp_lib.utils import set_random_seed
EPS = 1e-15

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_layers, num_nodes, with_feat=False, sparse=False):
        super(GCN, self).__init__()
        self.with_feat = with_feat
        if not self.with_feat:
            self.embedding = Embedding(num_nodes, embed_dim, sparse=sparse)
        self.gcs = nn.ModuleList()
        self.gcs.append(GCNConv(in_channels, out_channels))
        for _ in range(num_layers-1):
            self.gcs.append(GCNConv(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for gc in self.gcs:
            reset(gc)
        self.embedding.reset_parameters()

    def forward(self, x, edge_index):
        if not self.with_feat:
            x = self.embedding(x)
        for gc in self.gcs[:-1]:
            x = gc(x, edge_index).relu()
        x = gc(x, edge_index)
        x = torch.nn.functional.normalize(x)
        return x

    def recon_loss(self, z, pos_edge_index, decoder, neg_edge_index=None):

        pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)
