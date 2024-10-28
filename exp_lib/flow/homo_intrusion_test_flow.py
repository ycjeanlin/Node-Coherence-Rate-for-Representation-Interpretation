import os
import pandas as pd
import numpy as np
import torch

from exp_lib.utils import get_logger
from exp_lib.utils.coherence_test import CoherenceTest

logger = get_logger()

class HomoIntrusionTestFlow():
    def __init__(self, base_folder, dataset_name, data_suffix, topk, num_intruders, num_buckets, num_workers, num_processes) -> None:
        self.base_folder = base_folder
        self.dataset_name = dataset_name
        self.data_suffix = data_suffix
        self.topk = topk
        self.num_intruders = num_intruders
        self.num_buckets = num_buckets
        self.num_workers = num_workers
        self.num_processes = num_processes
        logger.info("Initialize coherence tests")
        self.coherence_test = CoherenceTest(
            num_buckets=self.num_buckets, 
            topk=self.topk, 
            num_intrusion=self.num_intruders,
            num_workers=self.num_workers,
            num_processes=self.num_processes,
        )

    def gen_test(self, relation, sim_threshold, train_data):
        train_data = train_data.cpu()

        logger.info(f"Generate coherence test for {relation}...")
        os.makedirs(f"{self.base_folder}/{self.dataset_name}/coherence_test/{relation}", exist_ok=True)
        test_stats = self.coherence_test.gen_test(
            data=train_data,
            relation=relation,
            sim_threshold=sim_threshold,
            output_folder=f"{self.base_folder}/{self.dataset_name}/coherence_test/{relation}", 
            data_suffix=self.data_suffix
        )
        data_df = pd.DataFrame.from_dict(test_stats)
        data_df.to_csv(f"{self.base_folder}/{self.dataset_name}/exp/test_stats_{self.dataset_name}_{self.data_suffix}_{relation}.csv")


    def run_test(self, relation, model_name, node_embed, coherence_epsilon):
        test_sets = self.coherence_test.load_test(
            f"{self.base_folder}/{self.dataset_name}/coherence_test/{relation}", 
            self.data_suffix
        )
        dist_success_rate = []
        rank_success_rate = []
        bottmk = []
        num_tests = []
        models = []
        logger.info("Start testing")
        for b in test_sets:
            test_set = test_sets[b]
            if len(test_set)>0:
                coherence_rate, ranking_pass_rate = self.coherence_test.testing(test_set, node_embed, coherence_epsilon)
                dist_success_rate.append(coherence_rate)
                rank_success_rate.append(ranking_pass_rate)
                bottmk.append(b)
                num_tests.append(len(test_set))
                models.append(model_name)
        data_df = pd.DataFrame.from_dict({
            "Bottom K": bottmk,
            "Group Coherence Rate": dist_success_rate,
            "Rank Coherence Rate": rank_success_rate,
            "Number Tests": num_tests,
            "Model": models
        })

        return data_df

    