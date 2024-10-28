import os
import time
import glob
import pickle
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity

from multiprocessing import Pool

from exp_lib.utils import get_logger
from .utils import gen_coherence_test, merge_test_batch

from exp_lib.utils import get_logger
logger = get_logger()
share_data = []

def worker(sim_threshold, query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    for q in query_nodes:
        coherence_tests = random_dist(sim_threshold, q, topk, num_intruders, num_buckets)
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    return (output_folder, result)


def on_completion(result):
    output_folder = result[0]
    pickle.dump(result[1], open(f"{output_folder}/tmp/random_dist_{result[1][0][0]}.pkl", "wb"))
    logger.info(f"Node {result[1][0][0]} group completed")
    del result


def random_dist(sim_threshold, query_node, topk, num_intruders, num_buckets):
    # label distribution calculation
    sim_mx = share_data[0]
    sim = sim_mx[query_node]
    sim[query_node] = 0

    coherence_test = gen_coherence_test(sim, sim_threshold, query_node, topk, num_intruders, num_buckets)
    return coherence_test


def random_dist_coherence_test(
    relation,
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
    
    num_nodes = data.num_nodes
    node_random = torch.rand(num_nodes, 128)
    
    mx = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes) 
    adj_mx = mx
    for _ in range(hop-1):
        adj_mx = adj_mx @ mx
    dist_mx = (adj_mx @ node_random)

    norm = dist_mx.sum(axis=1).reshape(-1,1)
    norm[norm<=0] = 1.0
    dist_mx = dist_mx/norm
    sim_mx = cosine_similarity(dist_mx, dist_mx)
    del norm
    del adj_mx
    del dist_mx 

    global share_data
    if len(share_data)==0:
        share_data.append(sim_mx)
    else:
        share_data[0] = sim_mx
    pickle.dump(sim_mx, open(f"{output_folder}/{relation}.pkl", "wb"), protocol=4)

    queries = []
    batch_size = int(num_nodes/num_workers) + 1
    for b in range(num_workers):
        start_idx = b*batch_size
        end_idx = min(b*batch_size + batch_size, num_nodes)
        if start_idx < end_idx:
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



