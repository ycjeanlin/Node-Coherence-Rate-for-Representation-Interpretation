import torch
import torch.nn as nn
from torch.nn import Embedding
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling

from exp_lib.model.decoder import LinkDecoder
from exp_lib.flow.utils import reset
EPS = 1e-15


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_layers, num_nodes, p=0.5, with_target=False, with_feat=False, sparse=False):
        super(GCN, self).__init__()
        self.with_feat = with_feat
        self.with_target = with_target
        if not self.with_feat:
            self.embedding = Embedding(num_nodes, embed_dim, sparse=sparse)
        if self.with_target:
            in_channels = in_channels + out_channels
        
        self.decoder = LinkDecoder()
        self.gcs = nn.ModuleList()
        if num_layers>1:
            self.gcs.append(GCNConv(in_channels, embed_dim, cached=True))
            for _ in range(num_layers-2):
                self.gcs.append(GCNConv(embed_dim, embed_dim, cached=True))
            self.gcs.append(GCNConv(embed_dim, out_channels, cached=True))
        else:
            self.gcs.append(GCNConv(in_channels, out_channels, cached=True))
        self.dropout = nn.Dropout(p)
        self.act = nn.PReLU(embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def load_pretrain(self, embed_file):
        embeddings = torch.load(embed_file)
        self.embedding = Embedding.from_pretrained(embeddings=embeddings)
        self.with_feat = False

    def reset_parameters(self):
        for gc in self.gcs:
            reset(gc)
        if not self.with_feat:
            self.embedding.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, y=None):
        if not self.with_feat:
            x = self.embedding.weight
        if self.with_target:
            x = torch.cat((x,y), dim=-1)
        
        for gc in self.gcs[:-1]:
            x = self.dropout(x)
            x = gc(x, edge_index, edge_weight)
            x = self.act(x)
            
        x = self.dropout(x)
        x = self.gcs[-1](x, edge_index, edge_weight)
        x = torch.nn.functional.normalize(x)
        return x

    def embed(self, x, edge_index, y=None):
        with torch.no_grad():
            if not self.with_feat:
                x = self.embedding.weight
            if self.with_target:
                x = torch.cat((x,y), dim=-1)
            
            for gc in self.gcs[:-1]:
                x = gc(x, edge_index)
                x = self.act(x)
                
            x = self.gcs[-1](x, edge_index)
            # x = torch.nn.functional.normalize(x)
            return x

    def predict(self, x, edge_index):
        z = self.forward(x, edge_index)
        y = self.softmax(z)
        return y


    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        decoder = LinkDecoder()
        pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - decoder(z, neg_edge_index, sigmoid=True) + EPS
        ).mean()
        return pos_loss + neg_loss


