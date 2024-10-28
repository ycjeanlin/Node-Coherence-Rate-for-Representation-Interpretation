import torch
import os
import numpy as np
import pickle
from exp_lib.utils import get_logger
logger = get_logger()


def gen_coherence_test(
        sim, 
        sim_threshold, 
        query_node, 
        topk, 
        num_intruders,
        num_buckets
    ):
    coherence_tests = []
    if len(sim[sim>0])<=topk+num_intruders:
        return coherence_tests

    p_threshold = np.quantile(sim[sim>0], sim_threshold)
    n_threshold = np.quantile(sim[sim>0], 0.05)
    # topk_threshold = np.quantile(sim[sim>0], 0.95)
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
   

    intruders = ((0<sim)&(sim<=n_threshold)).nonzero()[0]
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


def hete_gen_coherence_test(
        sim, 
        sim_threshold, 
        query_node, 
        target_nodes, 
        topk, 
        num_intruders, 
        num_buckets
    ):
    coherence_tests = []
    target_sim = sim[target_nodes]
    if len(sim[sim>0])<=topk+num_intruders:
        return coherence_tests
        
    p_threshold = np.quantile(target_sim[target_sim>0], sim_threshold)
    n_threshold = np.quantile(target_sim[target_sim>0], 0.05)
    sim = sim[target_nodes]
    similar_batch = []
    lb = sim[sim>0].max()
    for i in range(topk):
        ub = lb 
        if i==topk-1:
            lb = p_threshold
            batch = ((ub>=sim)&(sim>=lb)).nonzero()[0]
        else:
            lb = (ub - p_threshold)/2 + p_threshold 
            if ub==lb:
                lb = ub - 1e-5
            batch = ((ub>=sim)&(sim>lb)).nonzero()[0]
        
        if lb>n_threshold and len(batch)>0:
            similar_batch.append(batch)
        else:
            similar_batch.append([-1]) 
    
    intruders = ((0<sim)&(sim<=n_threshold)).nonzero()[0]
    for b in range(num_buckets):
        coherence_test = np.zeros(2 + topk + num_intruders)
        coherence_test[0] = query_node
        coherence_test[-1] = b
        for i in range(1, 1+topk):
            np.random.shuffle(similar_batch[-i])
            if similar_batch[-i][0]>=0:
                coherence_test[i:1+i] = target_nodes[similar_batch[-i][0]]
            else:
                coherence_test[i:1+i] = -1 
            

        if len(intruders) > 0:
            sample_nidxs = np.random.choice(intruders, size=num_intruders)
            coherence_test[1+topk:1+topk+num_intruders] = target_nodes[sample_nidxs]
        else:
            coherence_test[1+topk:1+topk+num_intruders] = -1
        coherence_tests.append(coherence_test)
                    
    return coherence_tests


def merge_test_batch(num_nodes, topk, num_intruders, num_buckets, output_folder, nidx2idx=None):
    tmp_folder = f"{output_folder}/tmp"
    batch_tests = {}
    for b in range(num_buckets):
        batch_tests[b] = []

    test_stats = {}
    test_stats['num_tests'] = 0
    similar_coverage = np.zeros(num_nodes)
    intruder_coverage = np.zeros(num_nodes)
    no_test_nodes = 0
    for f in os.listdir(tmp_folder):
        result = pickle.load(open(f"{tmp_folder}/{f}", "rb"))
        for intrusion_test in result:
            intruders = intrusion_test[1+topk:1+topk+num_intruders]
            if not (intruders<0).any():
                similars = intrusion_test[1:topk+1]
                b = int(intrusion_test[-1])
                batch_tests[b].append(intrusion_test[:-1])
                if nidx2idx!=None:
                    similars = np.array([nidx2idx[n] for n in similars if n>=0])
                    intruders = np.array([nidx2idx[n] for n in intruders if n>=0])
                else:
                    similars = similars[similars>=0].astype(int)
                    intruders = intruders[intruders>=0].astype(int)
                if len(similars)>0:
                    similar_coverage[similars] = 1
                else: 
                    no_test_nodes += 1
                if len(intruders)>0:
                    intruder_coverage[intruders] = 1
                test_stats['num_tests'] += 1
    logger.info(f"No test nodes: {no_test_nodes}")
    test_stats["similar_node_coverage"] = [similar_coverage.mean()]
    test_stats["intruder_node_coverage"] =  [intruder_coverage.mean()]
    return batch_tests, test_stats
