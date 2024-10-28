import os
import time
import glob
import pickle
import torch
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
from multiprocessing import Pool

# from gis.common.logger import get_logger
# logger = get_logger(__name__)

from exp_lib.utils import get_logger
from .utils import gen_coherence_test, merge_test_batch
logger = get_logger()

graph = []


def worker(sim_threshold, query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    sim_list = {}
    for q in query_nodes:
        coherence_tests, sim = graph_distance(sim_threshold, q, topk, num_intruders, num_buckets)
        sim_list[q] = sim
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    if len(result)==0:
        print(query_nodes[:10])
    return (output_folder, result, sim_list)

def on_completion(result):
    output_folder = result[0]
    pickle.dump(result[1], open(f"{output_folder}/tmp/g_dist_{result[1][0][0]}.pkl", "wb"))
    pickle.dump(result[2], open(f"{output_folder}/score/g_dist_{result[1][0][0]}.pkl", "wb"))
    logger.info(f"Node {result[1][0][0]} group completed")
    del result


def graph_distance(sim_threshold, query_node, topk, num_intruders, num_buckets):
    g = graph[0]
    path_length = nx.single_source_shortest_path_length(g, query_node)
    g_dist = np.zeros(g.number_of_nodes())-1
    for v in path_length:
        g_dist[v] = path_length[v]
    diameter = g_dist.max()
    g_dist[g_dist==-1]=diameter+1
    sim = 1 -  g_dist/(diameter+1)
    sim[query_node] = 0

    coherence_test = gen_coherence_test(sim, sim_threshold, query_node, topk, num_intruders, num_buckets)
    return coherence_test, sim


def graph_distance_coherence_test(
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

    if len(graph)==0:
        graph.append(to_networkx(data))
    else:
        graph[0] = to_networkx(data)

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
    merge_g_dist_batch(num_nodes, output_folder)
  
    return batch_tests, test_stats

def merge_g_dist_batch(query_nodes, output_folder):
    tmp_folder = f"{output_folder}/score"
    sim_mx = np.zeros((query_nodes, query_nodes))

    for f in os.listdir(tmp_folder):
        result = pickle.load(open(f"{tmp_folder}/{f}", "rb"))
        for s, sim in result.items():
            sim_mx[s] = sim

    pickle.dump(sim_mx, open(f"{output_folder}/g_dist.pkl", "wb"))
    