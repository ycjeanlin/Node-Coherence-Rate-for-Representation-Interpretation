from json import decoder
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize, scale

from exp_lib.utils import get_logger

logger = get_logger()



class EVDFlow():
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def run(self, sim_mx, device):
        sim_mx = torch.Tensor(sim_mx).to(device)
        U, S, Vh = torch.linalg.svd(sim_mx)
        sorted_vidxs = torch.argsort(S, descending=True)
        sorted_U = U[:, sorted_vidxs]
        sorted_S = torch.sqrt(torch.diag(S[sorted_vidxs]))
        z = sorted_U[:,:self.embed_dim] @ sorted_S[:self.embed_dim][:, :self.embed_dim]

        return z
