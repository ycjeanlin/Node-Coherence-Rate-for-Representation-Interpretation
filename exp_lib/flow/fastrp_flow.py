import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, coalesce
from scipy.sparse import csr_matrix, csc_matrix, spdiags
from sklearn.metrics import roc_auc_score 
from sklearn.preprocessing import normalize, scale
from sklearn import random_projection

from exp_lib.utils import get_logger, cal_rank

logger = get_logger()


class FastRPFlow():
    def __init__(self, embed_dim, weight1, weight2, weight3, weight4, projection_method, input_matrix, alpha, normalization):
        self.embed_dim = embed_dim
        self.weights = [weight1, weight2, weight3, weight4]
        self.projection_method = projection_method
        self.input_matrix = input_matrix
        self.alpha = alpha
        self.normalization = normalization

    def run(self, x, train_data, device, checkpoint_embed_file, patience):
        edge_index, _ = add_self_loops(train_data.edge_index)
        A = to_scipy_sparse_matrix(edge_index, num_nodes=train_data.num_nodes)
        D = A.sum(axis=-1)
        D[D==0] = 1
        A_hat = A/D
        A_hat = csc_matrix(A_hat)

        U_list = self.fastrp_projection(
            A_hat,
            q=len(self.weights),
            dim=self.embed_dim,
            projection_method=self.projection_method,
            input_matrix=self.input_matrix,
            alpha=self.alpha,
        )
        U = self.fastrp_merge(U_list, self.weights, self.normalization)
        z = self.output_embed(U).to(device)
        z = F.normalize(z)

        return z

    def output_embed(self, U):
        z = torch.Tensor(U)
        return z
    
    def fastrp_projection(self, A, q=3, dim=128, projection_method='gaussian', input_matrix='adj', alpha=None):
        assert input_matrix == 'adj' or input_matrix == 'trans'
        assert projection_method == 'gaussian' or projection_method == 'sparse'
        
        N = A.shape[0]
        if input_matrix == 'adj':
            M = A
        else:
            normalizer = spdiags(np.squeeze(1.0 / csc_matrix.sum(A, axis=1) ), 0, N, N)
            M = normalizer @ A
        # Gaussian projection matrix
        if projection_method == 'gaussian':
            transformer = random_projection.GaussianRandomProjection(n_components=dim, random_state=42)
        # Sparse projection matrix
        else:
            transformer = random_projection.SparseRandomProjection(n_components=dim, random_state=42)
        Y = transformer.fit(M)
        # Random projection for A
        if alpha is not None:
            Y.components_ = Y.components_ @ spdiags( \
                            np.squeeze(np.power(csc_matrix.sum(A, axis=1), alpha)), 0, N, N)
        cur_U = transformer.transform(M)
        U_list = [cur_U]
        
        for i in range(2, q + 1):
            cur_U = M @ cur_U
            U_list.append(cur_U)
        return U_list

    def fastrp_merge(self, U_list, weights, normalization=False):
        dense_U_list = [_U.todense() for _U in U_list] if type(U_list[0]) == csc_matrix else U_list
        _U_list = [normalize(_U, norm='l2', axis=1) for _U in dense_U_list] if normalization else dense_U_list

        if weights is None:
            return np.concatenate(_U_list, axis=1)
        U = np.zeros_like(_U_list[0])
        for cur_U, weight in zip(_U_list, weights):
            U += cur_U * weight
        # U = scale(U.todense())
        # U = normalize(U.todense(), norm='l2', axis=1)
        return scale(U.todense()) if type(U) == csr_matrix else scale(U)
