import numpy as np
import pickle
import torch
from exp_lib.semantic_relation import (
    attr_coherence_test,
    label_coherence_test,
    attr_dist_coherence_test,
    random_dist_coherence_test,
    label_dist_coherence_test,
    has_link_coherence_test,
    common_neighbor_coherence_test,
    degree_dist_coherence_test,
    graph_distance_coherence_test,
    pagerank_coherence_test,
)
from exp_lib.utils.embed_distance import embed_dist
from exp_lib.utils import get_logger


logger = get_logger()



class CoherenceTest():
    def __init__(self, 
        num_buckets,
        num_workers, 
        num_processes, 
        topk=3, 
        num_intrusion=1
    ):
        self.topk = topk
        self.num_intrusion= num_intrusion
        self.num_buckets = num_buckets
        self.num_workers = num_workers
        self.num_processes= num_processes

    def gen_test(self, data, relation, sim_threshold, output_folder, data_suffix):
        if relation=="label":
            batch_tests, test_stats = label_coherence_test(
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="attr":
            batch_tests, test_stats = attr_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="attr_dist":
            batch_tests, test_stats = attr_dist_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="label_dist":
            batch_tests, test_stats = label_dist_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="has_link":
            batch_tests, test_stats = has_link_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="common_neighbor":
            batch_tests, test_stats = common_neighbor_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="degree_dist":
            batch_tests, test_stats = degree_dist_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="pagerank":
            batch_tests, test_stats = pagerank_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif relation=="g_dist":
            batch_tests, test_stats = graph_distance_coherence_test(
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        else:
            batch_tests, test_stats = random_dist_coherence_test(
                relation,
                sim_threshold,
                data,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
            
        for b in batch_tests:
            final_test_set = batch_tests[b]
            if data_suffix!="baseline" :
                self.output_test(np.array(final_test_set), f"{output_folder}/{data_suffix}_b{b}.csv")
            else:
                self.output_test(np.array(final_test_set), f"{output_folder}/test_b{b}.csv")
        return test_stats

    def load_test(self, input_folder, data_suffix):
        test_sets = {}
        for b in range(self.num_buckets):
            if data_suffix!="baseline":
                test_set = np.loadtxt(f"{input_folder}/{data_suffix}_b{b}.csv", delimiter=",")
            else:
                test_set = np.loadtxt(f"{input_folder}/test_b{b}.csv", delimiter=",")
            test_sets[b] = test_set
        return test_sets

    def testing(self, test_set, node_embed, intrusion_epsilon):
        intrusion_passes = []
        rankings_passes = []
        for test in test_set:
            test = test.astype(int)
            similars = test[1:-self.num_intrusion]
            similars = similars[similars>=0]
            intruders = test[-self.num_intrusion:]
            intruders = intruders[intruders>=0]
            if len(similars)==self.topk and len(intruders)==self.num_intrusion:
                similar_dist = embed_dist(node_embed, test[0:1], similars)[0]
                foreign_dist =  embed_dist(node_embed, test[0:1], intruders)
                min_foreign_dist = foreign_dist.min()
                intrusion_passes.extend((min_foreign_dist-similar_dist)>=intrusion_epsilon)

                pass_test = True
                for i in range(1, self.topk):
                    pass_test = ((similar_dist[i-1]-similar_dist[i])>0)
                    if not pass_test:
                        break
                rankings_passes.append(pass_test)

        return (
            np.hstack(intrusion_passes).mean(), 
            np.array(rankings_passes).mean(), 
        )

    def output_test(self, test_set, filename):
        test_set = np.array(test_set)
        np.savetxt(filename, test_set, delimiter=",", fmt="%d")



class HeteCoherenceTest(CoherenceTest):
    def __init__(self, source_nodes, target_nodes, num_buckets, num_workers, num_processes, topk=3, num_intrusion=1):
        super().__init__(
            num_buckets, 
            num_workers, 
            num_processes, 
            topk, 
            num_intrusion
        )
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes

    def gen_test(self, 
        test_config,
        similarity, 
        sim_type, 
        sim_threshold, 
        data, 
        dataset,
        output_folder, 
        data_suffix, 
        source_type, 
        target_type
    ):
        if similarity=="label":
            raise NotImplementedError
        elif similarity=="label_dist":
            source_hops=test_config["first_reach"][source_type][sim_type]
            target_hops=test_config["first_reach"][target_type][sim_type]
            batch_tests, test_stats = hete_label_dist_coherence_test(
                sim_threshold,
                data,
                self.source_nodes,
                self.target_nodes,
                sim_type,
                dataset.node_slices[sim_type],
                source_hops,
                target_hops,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif similarity=="has_metapath":
            metapath = sim_type
            logger.info("Calculate metapath {}".format(metapath))
            dataset.get_metapath_graph(dataset.train_data, metapath)
            source_node_type = dataset.data.source_node_type 
            target_node_type = dataset.data.target_node_type 

            batch_tests, test_stats = hete_has_metapath_coherence_test(
                sim_threshold,
                dataset.data,
                self.source_nodes,
                self.target_nodes,
                sim_type,
                dataset.node_slices[source_node_type],
                dataset.node_slices[target_node_type],
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif similarity=="common_neighbor":
            source_hops=test_config["first_reach"][source_type][sim_type]
            target_hops=test_config["first_reach"][target_type][sim_type]
            batch_tests, test_stats = hete_common_neighbor_coherence_test(
                sim_threshold,
                data,
                self.source_nodes,
                self.target_nodes,
                sim_type,
                dataset.node_slices[sim_type],
                source_hops,
                target_hops,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif similarity=="degree_dist":
            source_hops=test_config["first_reach"][source_type][sim_type]
            target_hops=test_config["first_reach"][target_type][sim_type]
            batch_tests, test_stats = hete_degree_dist_coherence_test(
                sim_threshold,
                data,
                self.source_nodes,
                self.target_nodes,
                sim_type,
                dataset.node_slices[sim_type],
                source_hops,
                target_hops,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif similarity=="pagerank":
            batch_tests, test_stats = hete_pagerank_coherence_test(
                sim_threshold,
                data,
                self.source_nodes,
                self.target_nodes,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif similarity=="g_dist":
            batch_tests, test_stats = hete_graph_distance_coherence_test(
                sim_threshold,
                data,
                self.source_nodes,
                self.target_nodes,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
        elif similarity=="tfidf":
            raise NotImplementedError
        else:
            source_hops=test_config["first_reach"][source_type][sim_type]
            target_hops=test_config["first_reach"][target_type][sim_type]
            batch_tests, test_stats = hete_random_dist_coherence_test(
                similarity,
                sim_threshold,
                data,
                self.source_nodes,
                self.target_nodes,
                sim_type,
                dataset.node_slices[sim_type],
                source_hops,
                target_hops,
                self.topk,
                self.num_intrusion,
                self.num_buckets,
                output_folder,
                self.num_workers,
                self.num_processes,
            )
            
        for b in batch_tests:
            final_test_set = batch_tests[b]
            if data_suffix!="baseline" :
                self.output_test(np.array(final_test_set), f"{output_folder}/{data_suffix}_{source_type}_{target_type}_{sim_type}_b{b}.csv")
            else:
                self.output_test(np.array(final_test_set), f"{output_folder}/test_b{b}.csv")
        return test_stats

    def load_test(self, input_folder, data_suffix, sim_type, source_type, target_type):
        test_sets = {}
        for b in range(self.num_buckets):
            test_set = np.loadtxt(f"{input_folder}/{data_suffix}_{source_type}_{target_type}_{sim_type}_b{b}.csv", delimiter=",")
            test_sets[b] = test_set
        return test_sets