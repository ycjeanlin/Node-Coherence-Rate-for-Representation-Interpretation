import torch
import torch.nn as nn
from torch.nn import Embedding
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling
from exp_lib.model.decoder import LinkDecoder

from exp_lib.utils import set_random_seed
from exp_lib.flow.utils import reset
EPS = 1e-15


import torch

from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_layers, num_heads, num_nodes, with_feat=False, with_target=False, p=0.5, sparse=False, pretrain_file=None):
        super(GAT, self).__init__()
        self.with_feat = with_feat
        self.with_target = with_target
        if not self.with_feat:
            if pretrain_file:
                weight = torch.load(pretrain_file)
                self.embedding = nn.Embedding.from_pretrained(weight)
            else:
                self.embedding = Embedding(num_nodes, embed_dim, sparse=sparse)

        self.decoder = LinkDecoder()
        self.gcs = nn.ModuleList()
        if num_layers>1:
            self.gcs.append(GATConv(in_channels, embed_dim, heads=num_heads))
            for _ in range(num_layers-2):
                self.gcs.append(GATConv(num_heads * embed_dim, embed_dim, heads=num_heads))
            self.gcs.append(GATConv(num_heads * embed_dim, out_channels, heads=num_heads, concat=False))
        else:
            self.gcs.append(GATConv(in_channels, out_channels, heads=num_heads, concat=False))

        self.dropout = nn.Dropout(p)
        self.act= nn.ELU()
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        for gc in self.gcs:
            reset(gc)
        if not self.with_feat:
            self.embedding.reset_parameters()

    def forward(self, x, edge_index, y=None):
        if not self.with_feat:
            x = self.embedding(x)
        if self.with_target:
            x = torch.cat((x,y), dim=-1)
    
        for gc in self.gcs[:-1]:
            x = self.dropout(x)
            x = self.act(gc(x, edge_index))
            
        x = self.dropout(x)
        x = self.gcs[-1](x, edge_index)
        x = torch.nn.functional.normalize(x)
        return x
        
    def embed(self, x, edge_index, y=None):
        with torch.no_grad():
            if not self.with_feat:
                x = self.embedding(x)
            if self.with_target:
                x = torch.cat((x,y), dim=-1)
        
            for gc in self.gcs[:-1]:
                x = self.act(gc(x, edge_index))
                
            x = self.gcs[-1](x, edge_index)
            x = torch.nn.functional.normalize(x)
        return x

    def predict(self, x, edge_index):
        z = self.forward(x, edge_index)
        y = self.softmax(z)
        return y

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss



