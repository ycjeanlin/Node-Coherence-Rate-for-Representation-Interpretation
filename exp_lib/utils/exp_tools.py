import os
import json
import random
import glob
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Optional
from torch_scatter import scatter_min, scatter_sum
import torch.nn.functional as F
import scipy.stats as st

from torch_geometric.utils import to_undirected, coalesce, sort_edge_index

from exp_lib.utils.embed_distance import embed_dist


def degree(index, num_nodes,
           dtype: Optional[torch.dtype] = None):
    N = num_nodes
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(1), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index[0], one)

def set_random_seed(random_seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def plot_points(embed, dataset):
    data = dataset.data
    z = TSNE(n_components=2).fit_transform(embed)
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20)
    plt.axis('off')
    plt.show()

def get_rel_batch(base_folder, dataset_name, data_suffix, node_relations, num_nodes, num_batches, topk, device):
    rel_batches = {}
    for sim in node_relations:
        test_folder = f"{base_folder}/{dataset_name}/intrusion_test/{sim}"
        if data_suffix=="baseline":
            files = glob.glob(f"{test_folder}/test*") 
        else:
            files = glob.glob(f"{test_folder}/{data_suffix}*")
            
        rel_batch = (np.zeros((num_nodes, num_batches)) - 1).astype(int)
        rel_batch[:, 0] = np.arange(num_nodes)
        for b, f in enumerate(files):
            tests = np.loadtxt(f, delimiter=",").astype(int)
            query_nodes = tests[:,0]
            rel_batch[query_nodes, 1+b:1+(b+1)] = tests[:, topk:topk+1]
            # rel_batch[query_nodes, 1+b*num_allies:1+(b+1)*num_allies] = tests[:, 1:1+num_allies]
        rel_batches[sim] = torch.LongTensor(rel_batch).to(device)
    return rel_batches

def get_rel_edge_index(data, base_folder, dataset_name, data_suffix, node_relations, num_nodes, num_batches, topk, device):
    relation_weights = node_relations[data_suffix]
    neg_rel_weight = -np.array(list(relation_weights.values()))
    rel2idx = {rel: idx for idx, rel in enumerate(relation_weights.keys())}
    rel_batches = get_rel_batch(
        base_folder, 
        dataset_name, 
        data_suffix, 
        relation_weights, 
        num_nodes, 
        num_batches,
        topk,
        device
    )

    row = []
    col = []
    values = []
    for query_node in np.arange(data.num_nodes):
        pred_dict = {}
        for rel in relation_weights:
            sim_nodes = rel_batches[rel][query_node].cpu().numpy()[1:]
            for n in sim_nodes:
                if n == -1:
                    continue
                if n not in pred_dict:
                    pred_dict[n] = neg_rel_weight.copy()
                pred_dict[n][rel2idx[rel]] = relation_weights[rel]

        for n in pred_dict:
            value = pred_dict[n].sum()
            pos_prob = 1/(1+np.exp(-value))
            if pos_prob > node_relations["threshold"]:
                row.append(query_node)
                col.append(n)
                values.append(pos_prob)

    edge_index = torch.stack((torch.LongTensor(row), torch.LongTensor(col)))
    edge_attr = torch.Tensor(values) 
    edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr, reduce="max")
    edge_index, edge_attr = coalesce(edge_index, edge_attr)
    # edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
    # nidx = edge_index[0][0]
    # node_col = []
    # attr = [] 
    # max_num_edge = 3
    # new_col = []
    # new_row = []
    # new_attr = []
    # for idx, n in enumerate(edge_index[0]):
    #     if n != nidx:
    #         if len(node_col) <= max_num_edge:
    #             new_row.extend([nidx] * len(node_col))
    #             new_col.extend(node_col)
    #             new_attr.extend(attr)
    #         else:
    #             node_col = torch.LongTensor(node_col)
    #             attr = torch.Tensor(attr)
    #             sorted_idxs = torch.argsort(attr, descending=True)
    #             new_row.extend([nidx] * max_num_edge)
    #             new_col.extend(node_col[sorted_idxs[:max_num_edge]])
    #             new_attr.extend(attr[sorted_idxs[:max_num_edge]])
    #         nidx = n
    #         node_col = []
    #         attr = []   
    #     node_col.append(edge_index[1][idx])
    #     attr.append(edge_attr[idx])
    # edge_index = torch.stack((torch.LongTensor(new_row), torch.LongTensor(new_col)))
    # edge_attr = torch.Tensor(new_attr) 
    return edge_index.to(device), edge_attr.to(device)

def cal_rank(z, edge_index, target_nodes):
    sid = 0
    ranks = torch.zeros(edge_index.shape[1])
    z = F.normalize(z, dim=-1)
    eu_dist = torch.cdist(z,z)
    for i in range(edge_index.shape[1]):
        sid, tid = edge_index[:, i]
        yt = eu_dist[sid, tid]
        rank_t = (eu_dist[sid, target_nodes]<=yt).sum()
        ranks[i] = rank_t
    return ranks

def hit_rate(rank, row_index, topk):
    min_rank, _ = scatter_min(rank, row_index, dim=-1)
    min_rank = min_rank[min_rank!=0]
    return (min_rank<=topk).float().mean().item()

def mrr(rank, row_index):
    min_rank, _ = scatter_min(rank, row_index, dim=-1)
    rr = 1/min_rank[min_rank!=0]
    return (rr.mean()).item()

def precision_k(rank, row_index, topk):
    hits = (rank<=topk).float()
    total_nodes = len(row_index.unique())
    total_hit  = scatter_sum(hits, row_index)
    precisions = total_hit/topk
    return (precisions.sum()/total_nodes).item()

def recall_k(rank, row_index, topk):
    hits = (rank<=topk).float()
    total_nodes = len(row_index.unique())
    total_hit = scatter_sum(hits, row_index)
    total_targets = scatter_sum(torch.ones_like(row_index), row_index)
    total_targets[total_targets==0] = 1.0
    recalls = total_hit/total_targets
    return (recalls.sum()/total_nodes).item()

def random_test(z, topk, num_intruders, intrusion_decision_quantile, num_iters=30):
    intrusion_pass_rates = []
    ranking_pass_rates = []
    epsilons = []
    num_nodes = z.shape[0]
    for _ in range(num_iters):
        distance_diff = []
        intrusion_pass_test = []
        ranking_pass_test = []
        for u in range(num_nodes):
            rand_nodes = np.random.randint(0, num_nodes, topk+num_intruders)
            rand_dist = embed_dist(z, [u], rand_nodes)
            d1_mx = rand_dist.repeat(topk+num_intruders, axis=0)
            d2_mx = d1_mx.T
            delta_d = (d1_mx-d2_mx).reshape(-1)
            distance_diff.append(delta_d)
            intrusion_pass_test.append(rand_dist[0][-1]-rand_dist[0][:-1])
            ranking_pass_test.append(
                (rand_dist[0][0]>rand_dist[0][1]) & (rand_dist[0][1]>rand_dist[0][2])
            )
        distance_diff = np.hstack(distance_diff)
        diff_std = np.std(distance_diff)
        z_stat = st.norm.ppf(intrusion_decision_quantile)
        epsilon = diff_std * z_stat * 0.25
        # epsilon = np.quantile(distance_diff, q=intrusion_decision_quantile)
        epsilons.append(epsilon)
        ranking_pass_rates.append(np.array(ranking_pass_test).mean())
        intrusion_pass_rates.append((np.hstack(intrusion_pass_test)>=epsilon).mean())
    
    intrusion_rate_ub = np.mean(intrusion_pass_rates) + z_stat*np.std(intrusion_pass_rates)
    ranking_rate_ub = np.mean(ranking_pass_rates) + z_stat*np.std(ranking_pass_rates)
    # import pdb; pdb.set_trace()
    return np.array(epsilons).mean(), intrusion_rate_ub, ranking_rate_ub

def pass_rate_upper_bound(z, topk, intrusion_threshold):
    pair_dist = torch.cdist(z, z)
    avg_dist = pair_dist.mean()
    pair_dist = pair_dist.fill_diagonal_(avg_dist)
    var = ((pair_dist - pair_dist.mean())**2).sum()/(pair_dist.shape[0]**2-pair_dist.shape[0]-1)
    ub = (2 * var)/(2 * var + intrusion_threshold**2)
    # import pdb; pdb.set_trace()
    return (ub**topk).item()

    
def classifier_result(config, dataset_name):
    precisions = []
    recalls = []
    f1scores = []
    accuracies = []
    models = []
    exp_ids = []
    base_folder = f"./data/{dataset_name}"
    task = 'nc'
    for data_suffix in config[task]:
        for model in config[task][data_suffix]:
            for model_suffix in config[task][data_suffix][model]: 
                if model=="random":
                    continue
                for exp_id in range(10):
                    exp_file = f"{base_folder}/exp/{task}_{dataset_name}_{data_suffix}_{model}_{model_suffix}_{exp_id}.csv"
                    df = pd.read_csv(exp_file, index_col=0)
                    test_mask = df["test_mask"]
                    y_pred = df["y_pred"][test_mask]
                    y_true = df["y_true"][test_mask]
                    precision, recall, f1score, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="macro")
                    acc = accuracy_score(y_true, y_pred)
                    precisions.append(precision)
                    accuracies.append(acc)
                    recalls.append(recall)
                    f1scores.append(f1score)
                    exp_ids.append(exp_id)
                    if model!="baseline":
                        models.append(f"{model}_{model_suffix}")
                    else:
                        models.append(f"{model}")

    data_df = pd.DataFrame.from_dict({
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1-score": f1scores,
        "Model": models,
    })
    return data_df


def link_pred_result(config, dataset_name):
    task = "link"
    aucs = []
    mrrs = []
    hit_k5 = []
    hit_k10 = []
    precision_k5 = []
    precision_k10 = []
    recall_k5 = []
    recall_k10 = []
    models = []
    data_suffices = []
    base_folder = f"./data/{dataset_name}"
    for data_suffix in config[task]:
        for model in config[task][data_suffix]:
            for model_suffix in config[task][data_suffix][model]: 
                if model=="random":
                    continue
                print(f"{base_folder}/exp/{task}_{dataset_name}_{data_suffix}_{model}_{model_suffix}.csv")
                df = pd.read_csv(f"{base_folder}/exp/{task}_{dataset_name}_{data_suffix}_{model}_{model_suffix}.csv", index_col=0)
                test_mask = df["test_mask"]
                edge_mask = (test_mask & (df["y_true"]==1))
                y_score = df["y_score"][test_mask]
                y_true = df["y_true"][test_mask]
                rank = torch.Tensor(df["rank"][edge_mask].values)
                row_index = torch.LongTensor(df["row"][edge_mask].values)
                auc = roc_auc_score(y_true, y_score)
                mrrs.append(mrr(rank, row_index))
                hit_k5.append(hit_rate(rank, row_index, topk=5))
                hit_k10.append(hit_rate(rank, row_index, topk=10))
                precision_k5.append(precision_k(rank, row_index, topk=5))
                precision_k10.append(precision_k(rank, row_index, topk=10))
                recall_k5.append(recall_k(rank, row_index, topk=5))
                recall_k10.append(recall_k(rank, row_index, topk=10))
                aucs.append(auc)
                data_suffices.append(data_suffix)
                if model!="baseline":
                    models.append(f"{model}_{model_suffix}")
                else:
                    models.append(f"{model}")

    data_df = pd.DataFrame.from_dict({
        "AUC": aucs,
        "MRR": mrrs,
        "Hit@5": hit_k5,
        "Hit@10": hit_k10,
        "Precision@5": precision_k5,
        "Precision@10": precision_k10,
        "Recall@5": recall_k5,
        "Recall@10": recall_k10,
        "Model": models,
    })
    return data_df