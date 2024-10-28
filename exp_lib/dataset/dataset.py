import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomLinkSplit

from exp_lib.utils import set_random_seed

class HomoGraph():
    def __init__(self, root):
        self.base_folder = root

    def data_preprocess(self):
        eps = 2.2204e-16
        norm = self.data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        self.data.x = self.data.x.div(norm.expand_as(self.data.x))

    def make_transductive_data(self, num_train_per_class=20, num_val=500, val_seed=42, seed=42):
        set_random_seed(val_seed)
        data = self.data
        remaining = torch.arange(self.data.num_nodes)
        remaining = remaining[torch.randperm(remaining.size(0))]
        data.val_mask.fill_(False)
        data.val_mask[remaining[:num_val]] = True
        
        set_random_seed(seed)
        data.train_mask.fill_(False)
        remaining = remaining[num_val:]
        for c in range(self.dataset.num_classes):
            idx = (data.y[remaining] == c).nonzero(as_tuple=False).view(-1)
            idx = remaining[idx[torch.randperm(idx.size(0))[:num_train_per_class]]]
            data.train_mask[idx] = True

        data.test_mask.fill_(True)
        data.test_mask[data.train_mask] = False
        data.test_mask[data.val_mask] = False 
        
        self.data, self.slices = InMemoryDataset.collate([data])

    def make_classification_data(self, train_size=0.1, seed=42):
        set_random_seed(seed)
        self.data.train_mask.fill_(False)
        for j in range(self.dataset.num_classes):
            group = (self.data.y==j).nonzero().view(-1)
            perm_idxs = torch.randperm(len(group))
            num_train = int(len(group) * train_size)
            self.data.train_mask[group[perm_idxs[:num_train]]] = True
        
        self.data.val_mask.fill_(False)
        self.data.val_mask[self.data.train_mask] = True
        self.data.test_mask.fill_(True)
        self.data.test_mask[self.data.train_mask] = False 

    def make_link_data(self, val_size, test_size, seed=42):
        set_random_seed(seed)
        test_split = RandomLinkSplit(num_val=0, num_test=test_size, is_undirected=True)
        train_data, _, test_data = test_split(self.data)

        del train_data.edge_label
        del train_data.edge_label_index
        val_split = RandomLinkSplit(num_val=0, num_test=val_size, is_undirected=True)
        train_data, _, val_data = val_split(train_data)

        train_data.train_mask.fill_(True)
        train_data.val_mask.fill_(True)
        train_data.test_mask.fill_(False)
        
        return train_data, val_data, test_data


class HeteGraph():
    def __init__(self, root):
        self.base_folder = root
    
    def make_classification_data(self, target_nodes, train_size, seed=42):
        set_random_seed(seed)
        train_idxs, test_idxs = train_test_split(
            target_nodes, 
            test_size=1-train_size, 
            stratify=self.data.y.numpy(),
            random_state=seed,
        )
        # train_idxs, valid_idxs = train_test_split(
        #     train_idxs, 
        #     test_size=val_size, 
        #     stratify=self.data.y[train_idxs].numpy(),
        #     random_state=seed,
        # )
        num_nodes = self.data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idxs] = True
        val_mask[train_idxs] = True
        test_mask[test_idxs] = True
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    def make_link_data(self, val_size, test_size, seed=42):
        set_random_seed(seed)
        test_split = RandomLinkSplit(
            num_val=0, 
            num_test=test_size, 
            is_undirected=True, 
            edge_types=self.edge_type,
            rev_edge_types=self.rev_edge_type
        )
        val_split = RandomLinkSplit(
            num_val=0, 
            num_test=val_size, 
            is_undirected=True, 
            edge_types=self.edge_type,
            rev_edge_types=self.rev_edge_type
        )
        self.train_data, _, self.test_data = test_split(self.dataset._data)
        del self.train_data[self.edge_type].edge_label
        del self.train_data[self.edge_type].edge_label_index
        
        self.data, _, self.val_data = val_split(self.train_data)

        if self.metapath=="homo":
            if self.edge_type is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to"
                    "be specified when operating on 'HeteroData' objects")

            homo_train_data = self.data.to_homogeneous(dummy_values=False)
            homo_train_data["edge_label"] = self.data[self.edge_type].edge_label
            offset_row = self.node_slices[self.edge_type[0]][0]
            offset_col = self.node_slices[self.edge_type[2]][0]
            row = self.data[self.edge_type].edge_label_index[0] + offset_row
            col = self.data[self.edge_type].edge_label_index[1] + offset_col
            homo_train_data["edge_label_index"] = torch.stack((row, col), dim=0)
            homo_train_data.num_nodes = homo_train_data.node_type.shape[0]
            homo_train_data.train_mask = torch.ones(homo_train_data.num_nodes).bool()
            homo_train_data.val_mask = torch.ones(homo_train_data.num_nodes).bool()
            homo_train_data.test_mask = torch.zeros(homo_train_data.num_nodes).bool()

            homo_test_data = self.test_data.to_homogeneous(dummy_values=False)
            homo_test_data["edge_label"] = self.test_data[self.edge_type].edge_label
            row = self.test_data[self.edge_type].edge_label_index[0] + offset_row
            col = self.test_data[self.edge_type].edge_label_index[1] + offset_col
            homo_test_data["edge_label_index"] = torch.stack((row, col), dim=0)
            homo_test_data.num_nodes = homo_test_data.node_type.shape[0]

            homo_val_data = self.val_data.to_homogeneous(dummy_values=False)
            homo_val_data["edge_label"] = self.val_data[self.edge_type].edge_label
            row = self.val_data[self.edge_type].edge_label_index[0] + offset_row
            col = self.val_data[self.edge_type].edge_label_index[1] + offset_col
            homo_val_data["edge_label_index"] = torch.stack((row, col), dim=0)
            homo_val_data.num_nodes = homo_val_data.node_type.shape[0]

            source_slice = self.node_slices[self.edge_type[0]]
            target_slice = self.node_slices[self.edge_type[2]]
            self.source_nodes = torch.arange(source_slice[0], source_slice[1])
            self.target_nodes = torch.arange(target_slice[0], target_slice[1])
            return homo_train_data, homo_val_data, homo_test_data
        else:
            return self.data, self.val_data, self.test_data