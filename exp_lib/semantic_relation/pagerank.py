import os
import time
import glob
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import scipy as sp
import scipy.sparse as sprs
from multiprocessing import Pool

# from gis.common.logger import get_logger
# logger = get_logger(__name__)

from exp_lib.utils import get_logger, degree
from .utils import gen_coherence_test, merge_test_batch
logger = get_logger()

share_data = []


def worker(sim_threshold, query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    sim_list = {}
    for q in query_nodes:
        coherence_tests, sim = pagerank_power(sim_threshold, q, topk, num_intruders, num_buckets)
        sim_list[q] = sim
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    return (output_folder, result, sim_list)

def on_completion(result):
    output_folder = result[0]
    pickle.dump(result[1], open(f"{output_folder}/tmp/pagerank_{result[1][0][0]}.pkl", "wb"))
    pickle.dump(result[2], open(f"{output_folder}/score/pagerank_{result[1][0][0]}.pkl", "wb"))
    logger.info(f"Node {result[1][0][0]} group completed")
    del result


def pagerank_power(sim_threshold, query_node, topk, num_intruders, num_buckets, alpha=0.85, max_iter=100, tol=1e-06, reverse=False):
    A = share_data[0]
    personalize = np.zeros(A.shape[0])
    personalize[:] = 0
    personalize[query_node] = 1.0
    if reverse:
        A = A.T

    n, _ = A.shape
    r = np.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, -1)
    s = (personalize / personalize.sum(axis=0, keepdims=True)) * n
 
    z_T = (((1 - alpha) * (r != 0) + (r == 0)) / n)[sp.newaxis, :]
    W = alpha * A.T @ D_1

    x = s
    oldx = np.zeros((n, 1))

    iteration = 0

    while sp.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    sim = (x / sum(x)).reshape(-1)
    sim[query_node] = 0

    coherence_test = gen_coherence_test(sim, sim_threshold, query_node, topk, num_intruders, num_buckets)
    return coherence_test, sim


def pagerank_coherence_test(
    sim_threshold,
    data, 
    topk, 
    num_intruders, 
    num_buckets, 
    output_folder, 
    num_workers, 
    num_processes,
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

    adj_mx = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    adj_mx = adj_mx.tocsr()
    global share_data
    if len(share_data)==0:
        share_data.append(adj_mx)
    else:
        share_data[0] = adj_mx

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
    merge_pagerank_batch(num_nodes, output_folder)
  
    return batch_tests, test_stats

def merge_pagerank_batch(num_nodes, output_folder):
    tmp_folder = f"{output_folder}/score"
    sim_mx = np.zeros((num_nodes, num_nodes))

    for f in os.listdir(tmp_folder):
        result = pickle.load(open(f"{tmp_folder}/{f}", "rb"))
        for s, sim in result.items():
            sim_mx[s] = sim

    pickle.dump(sim_mx, open(f"{output_folder}/pagerank.pkl", "wb"))
    