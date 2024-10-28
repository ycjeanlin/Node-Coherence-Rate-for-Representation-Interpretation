import torch


def embed_dist(z, query_node, target_nodes):
    dist = torch.cdist(z[query_node],  z[target_nodes])
    return dist.cpu().numpy()


