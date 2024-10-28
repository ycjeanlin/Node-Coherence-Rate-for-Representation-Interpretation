import os
import time
import glob
import pickle
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, add_remaining_self_loops, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from multiprocessing import Pool

from exp_lib.utils import get_logger
from .utils import merge_test_batch

logger = get_logger()
share_data = []

def worker(sim_threshold, query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    for q in query_nodes:
        coherence_tests = has_link(sim_threshold, q, topk, num_intruders, num_buckets)
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    return (output_folder, result)


def on_completion(result):
    output_folder = result[0]
    if len(result[1])>0:
        pickle.dump(result[1], open(f"{output_folder}/tmp/has_link_{result[1][0][0]}.pkl", "wb"))
        logger.info(f"Node {result[1][0][0]} group completed")
    del result


def has_link(sim_threshold, query_node, topk, num_intruders, num_buckets):
    sim_mx = share_data[0]
    sim = sim_mx[query_node]
    sim[query_node] = 0

    coherence_tests = []
    if len(sim[sim>0])<=topk+num_intruders:
        return coherence_tests

    # pos_values = np.unique(sim[sim>=sim_pos_threshold])
    # similar_batch = []
    # if len(pos_values)>=topk:
    #     sample_pos_values = np.random.choice(pos_values, size=topk, replace=False)
    #     sort_pos_values = np.sort(sample_pos_values)
    #     for i in range(topk):
    #         batch = (sim==sort_pos_values[i]).nonzero()[0]
    #         similar_batch.append(batch)
    
    p_threshold = sim[sim>0].min()
    n_threshold = 0 
    similar_batch = []
    lb = sim[sim>0].max()
    for i in range(topk):
        ub = lb 
        if i==topk-1:
            lb = p_threshold
            batch = ((ub>=sim)&(sim>=lb)).nonzero()[0]
        else:
            lb = (ub - p_threshold) / 2 + p_threshold
            if ub==lb:
                lb = ub - 1e-5
            batch = ((ub>=sim)&(sim>lb)).nonzero()[0]

        if lb>n_threshold and len(batch)>0:
            similar_batch.append(batch)
        else:
            similar_batch.append([-1]) 
   

    intruders = ((sim==0)).nonzero()[0]
    for b in range(num_buckets):
        coherence_test = np.zeros(2 + topk + num_intruders)
        coherence_test[0] = query_node
        coherence_test[-1] = b
        for i in range(1, 1+topk):
            np.random.shuffle(similar_batch[-i])
            coherence_test[i:1+i] = similar_batch[-i][0]
        
        if len(intruders) > 0:
            sample_nidxs = np.random.choice(intruders, size=num_intruders)
            coherence_test[1+topk:1+topk+num_intruders] = sample_nidxs
        else:
            coherence_test[1+topk:1+topk+num_intruders] = -1
        coherence_tests.append(coherence_test)

    return coherence_tests

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def has_link_coherence_test(
    sim_threshold,
    data, 
    topk, 
    num_intruders, 
    num_buckets, 
    output_folder, 
    num_workers, 
    num_processes,
    hop=1
):
    os.makedirs(f"{output_folder}/tmp", exist_ok=True)
    files = glob.glob(f"{output_folder}/tmp/*")
    for f in files:
        os.remove(f)

    os.makedirs(f"{output_folder}/score", exist_ok=True)
    files = glob.glob(f"{output_folder}/score/*")
    for f in files:
        os.remove(f)
    num_nodes = data.num_nodes

    logger.info("Normalize ...")

    logger.info("Edge index to adj mx ...")
    mx = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    adj_mx = mx
    for _ in range(hop-1):
        adj_mx = adj_mx @ mx
    edge_index, _ = from_scipy_sparse_matrix(adj_mx) 
    del adj_mx
    edge_index, edge_weight = gcn_norm(edge_index)
    sim_mx = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0].numpy()
    pickle.dump(sim_mx, open(f"{output_folder}/has_link.pkl", "wb"), protocol=4)
    

    global share_data
    if len(share_data)==0:
        share_data.append(sim_mx)
    else:
        share_data[0] = sim_mx

    queries = []
    batch_size = int(num_nodes/num_workers) + 1
    for b in range(num_workers):
        start_idx = b*batch_size
        end_idx = min(b*batch_size + batch_size, num_nodes)
        if start_idx < end_idx:
            # logger.info(f"{start_idx}, {end_idx}")
            queries.append(np.arange(start_idx, end_idx))

    # import pdb; pdb.set_trace()
    logger.info("Start calculate distance...")
    t0 = time.time()
    pool = Pool(num_processes)
    for query_nodes in queries:
        pool.apply_async(
            worker, 
            args=(
                sim_threshold,
                query_nodes, 
                topk, 
                num_intruders, 
                num_buckets, 
                output_folder
            ), 
            callback=on_completion
        )
    pool.close()
    pool.join() # wait for worker processes to exit
    diff_time = time.time() - t0
    logger.info(f"Time: {diff_time: .2f}")

    logger.info(f"Start to merge results")
    batch_tests, test_stats = merge_test_batch(num_nodes, topk, num_intruders, num_buckets, output_folder)
  
    return batch_tests, test_stats


