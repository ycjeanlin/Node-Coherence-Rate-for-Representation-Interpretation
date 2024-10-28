import torch
import numpy as np
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.datasets import Airports
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

from exp_lib.utils import set_random_seed
from .dataset import HomoGraph

class Airport(HomoGraph):
    def __init__(self, base_folder, airport):
        set_random_seed(42)
        self.base_folder = base_folder
        self.dataset = Airports(root=self.base_folder, name=airport)
        self.data = self.dataset._data
        self.data.train_mask=torch.zeros(self.data.num_nodes).bool()
        self.data.val_mask=torch.zeros(self.data.num_nodes).bool()
        self.data.test_mask=torch.zeros(self.data.num_nodes).bool()
        self.target_nodes = torch.arange(self.data.num_nodes)

    # def make_data(self):
    #     row = []
    #     col = []
    #     with open(f"{self.base_folder}/{self.edge_list}", "r") as fr:
    #         for line in fr:
    #             edge = line.strip().split()
    #             if edge[0] not in self.aid2nidx:
    #                 self.aid2nidx[edge[0]] = len(self.aid2nidx)
    #             if edge[1] not in self.aid2nidx:
    #                 self.aid2nidx[edge[1]] = len(self.aid2nidx)
    #             u = self.aid2nidx[edge[0]]
    #             v = self.aid2nidx[edge[1]]
    #             row.append(u)
    #             col.append(v)
    #             row.append(v)
    #             col.append(u)

    #     edge_index = torch.LongTensor([row, col])
    #     edge_index = remove_self_loops(edge_index)[0]
    #     edge_index = coalesce(edge_index)
    #     num_nodes = len(self.aid2nidx)

    #     y = torch.zeros(num_nodes)
    #     with open(f"{self.base_folder}/{self.label_file}", "r") as fr:
    #         for line in fr:
    #             if line.strip() != "node label":
    #                 label = line.strip().split()
    #                 u = self.aid2nidx[label[0]]
    #                 y[u] = int(label[1])


    #     self.data = Data(
    #         edge_index=edge_index, 
    #         y=y.long(),
    #         train_mask=torch.zeros(num_nodes).bool(),
    #         val_mask=torch.zeros(num_nodes).bool(),
    #         test_mask=torch.zeros(num_nodes).bool(),
    #     )
    #     self.data.num_nodes = num_nodes
    #     self.target_nodes = torch.arange(num_nodes)

