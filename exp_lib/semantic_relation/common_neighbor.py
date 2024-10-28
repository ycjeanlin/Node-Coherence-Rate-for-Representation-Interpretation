import os
import time
import glob
import pickle
import torch
import numpy as np

from multiprocessing import Pool
from torch_geometric.utils import to_scipy_sparse_matrix
from .utils import gen_coherence_test, merge_test_batch

from exp_lib.utils import get_logger

logger = get_logger()
share_data = []

def worker(sim_threshold, query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    for q in query_nodes:
        coherence_tests = common_neighbor(sim_threshold, q, topk, num_intruders, num_buckets)
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    return (output_folder, result)


def on_completion(result):
    output_folder = result[0]
    if len(result[1])>0:
        pickle.dump(result[1], open(f"{output_folder}/tmp/common_neighbor_{result[1][0][0]}.pkl", "wb"))
        logger.info(f"Node {result[1][0][0]} group completed")
    del result


def common_neighbor(sim_threshold, query_node, topk, num_intruders, num_buckets):
    sim_mx = share_data[0]
    sim = sim_mx[query_node]
    sim[query_node] = 0

    coherence_test = gen_coherence_test(sim, sim_threshold, query_node, topk, num_intruders, num_buckets)
    return coherence_test


def common_neighbor_coherence_test(
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

    logger.info("Calculate intersection...")
    mx = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    adj_mx = mx
    for _ in range(hop-1):
        adj_mx = adj_mx @ mx
    adj_mx = adj_mx.toarray()
    adj_mx[adj_mx>0] = 1
    adj_mx_t = adj_mx.T
    intersection = np.matmul(adj_mx, adj_mx_t)
    logger.info("Calculate union...")
    all = np.matmul(adj_mx.sum(axis=-1).reshape(-1,1), np.ones(data.num_nodes).reshape(1,-1)) +\
          np.matmul(np.ones(data.num_nodes).reshape(-1,1), adj_mx_t.sum(axis=0).reshape(1,-1))
    del adj_mx
    del adj_mx_t
    union = all - intersection
    del all
    logger.info("Calculate sim...")
    union[union==0] = 1.0
    sim_mx = intersection / union
    pickle.dump(sim_mx, open(f"{output_folder}/common_neighbor.pkl", "wb"), protocol=4)

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

