from torch_geometric.datasets import Planetoid

from .dataset import HomoGraph

class CiteSeer(HomoGraph):
    def __init__(self, root):
        super().__init__(root)
        self.dataset = Planetoid(root=root, name="CiteSeer") 
        self.data = self.dataset._data
        self.data_preprocess()
        self.target_nodes = (self.data.train_mask + self.data.test_mask + self.data.val_mask).nonzero().view(-1)
