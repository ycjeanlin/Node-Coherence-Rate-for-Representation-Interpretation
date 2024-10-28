import os
import time
import glob
import pickle
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity

from multiprocessing import Pool

from exp_lib.utils import degree, get_logger
from .utils import gen_coherence_test, merge_test_batch

logger = get_logger()
share_memory = []

def worker(sim_threshold, query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    sim_list = {}
    for q in query_nodes:
        coherence_tests, sim = degree_dist(sim_threshold, q, topk, num_intruders, num_buckets)
        sim_list[q] = sim
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    return (output_folder, result, sim_list)


def on_completion(result):
    output_folder = result[0]
    pickle.dump(result[1], open(f"{output_folder}/tmp/degree_dist_{result[1][0][0]}.pkl", "wb"))
    pickle.dump(result[2], open(f"{output_folder}/score/degree_dist_{result[1][0][0]}.pkl", "wb"))
    logger.info(f"Node {result[1][0][0]} group completed")
    del result


def degree_dist(sim_threshold, query_node, topk, num_intruders, num_buckets):
    # degree distribution calculation
    sim_mx = share_memory[0]
    sim = sim_mx[query_node]
    sim[query_node] = 0

    coherence_test = gen_coherence_test(sim, sim_threshold, query_node, topk, num_intruders, num_buckets)
    return coherence_test, sim


def degree_dist_coherence_test(
    sim_threshold,
    data, 
    topk, 
    num_intruders, 
    num_buckets, 
    output_folder, 
    num_workers, 
    num_processes,
    hop = 1
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
    node_degree = degree(data.edge_index, num_nodes).long()
    y_one_hot = torch.nn.functional.one_hot(node_degree).float().numpy()
    
    mx = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes) 
    adj_mx = mx
    for _ in range(hop-1):
        adj_mx = adj_mx @ mx
    
    dist_mx = (adj_mx @ y_one_hot)
    del mx
    del adj_mx
    del y_one_hot

    norm = dist_mx.sum(axis=1).reshape(-1,1)
    norm[norm<=0] = 1.0
    dist_mx = dist_mx/norm
    sim_mx = cosine_similarity(dist_mx, dist_mx)
    global share_memory
    if len(share_memory)==0:
        share_memory.append(sim_mx)
    else:
        share_memory[0] = sim_mx
    del norm
    del dist_mx 

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
    merge_degree_dist_batch(num_nodes, output_folder)
  
    return batch_tests, test_stats

def merge_degree_dist_batch(query_nodes, output_folder):
    tmp_folder = f"{output_folder}/score"
    sim_mx = np.zeros((query_nodes, query_nodes))

    for f in os.listdir(tmp_folder):
        result = pickle.load(open(f"{tmp_folder}/{f}", "rb"))
        for s, sim in result.items():
            sim_mx[s] = sim

    pickle.dump(sim_mx, open(f"{output_folder}/degree_dist.pkl", "wb"))
    


