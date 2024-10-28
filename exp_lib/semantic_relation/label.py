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
from .utils import merge_test_batch


from exp_lib.utils import get_logger
logger = get_logger()
share_data = []

def worker(query_nodes, topk, num_intruders, num_buckets, output_folder):
    result = []
    for q in query_nodes:
        coherence_tests = label(q, topk, num_intruders, num_buckets)
        if len(coherence_tests)>0:
            result.extend(coherence_tests)
    return (output_folder, result)


def on_completion(result):
    output_folder = result[0]
    pickle.dump(result[1], open(f"{output_folder}/tmp/label_{result[1][0][0]}.pkl", "wb"))
    logger.info(f"Node {result[1][0][0]} group completed")
    del result


def label(query_node, topk, num_intruders, num_buckets):
    # label distribution calculation
    sim_mx = share_data[0]
    sim = sim_mx[query_node]
    sim[query_node] = 0

    coherence_tests = []
    if len(sim[sim>0])<=topk+num_intruders:
        return coherence_tests

    similar_batch = (sim>0).nonzero()[0]
   
    # topk_nidxs = (sim>=topk_threshold).nonzero()[0]
    if len(similar_batch)>=topk:
        intruders = (sim==0).nonzero()[0]
        for b in range(num_buckets):
            coherence_test = np.zeros(2 + topk + num_intruders)
            coherence_test[0] = query_node
            coherence_test[-1] = b
            np.random.shuffle(similar_batch)
            for i in range(1, 1+topk):
                coherence_test[i:1+i] = similar_batch[i]
            
            if len(intruders) > 0:
                sample_nidxs = np.random.choice(intruders, size=num_intruders)
                coherence_test[1+topk:1+topk+num_intruders] = sample_nidxs
            else:
                coherence_test[1+topk:1+topk+num_intruders] = -1
            coherence_tests.append(coherence_test)
    return coherence_tests


def label_coherence_test(
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
    
    num_nodes = data.num_nodes
    node_label = data.y
    y_one_hot = torch.nn.functional.one_hot(node_label).float().numpy()
    sim_mx = (y_one_hot @ y_one_hot.T)

    global share_data
    if len(share_data)==0:
        share_data.append(sim_mx)
    else:
        share_data[0] = sim_mx
    pickle.dump(sim_mx, open(f"{output_folder}/label.pkl", "wb"), protocol=4)

    queries = []
    batch_size = int(num_nodes/num_workers) + 1
    for b in range(num_workers):
        start_idx = b*batch_size
        end_idx = min(b*batch_size + batch_size, num_nodes)
        if start_idx < end_idx:
            queries.append(np.arange(start_idx, end_idx))

    # import pdb; pdb.set_trace()
    logger.info("Start calculate distance...")
    t0 = time.time()
    pool = Pool(num_processes)
    for query_nodes in queries:
        pool.apply_async(
            worker, 
            args=(
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



