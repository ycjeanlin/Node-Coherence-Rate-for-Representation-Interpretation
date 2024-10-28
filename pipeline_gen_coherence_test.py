import os
import time
import json
import argparse
import torch

from exp_lib.utils import set_random_seed, get_logger
from exp_lib.dataset import load_homo_data
from exp_lib.flow import HomoIntrusionTestFlow


def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    parser.add_argument('--base-folder', type=str)
    parser.add_argument('--model-suffix', type=str, default="baseline")
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--task', type=str, help='method to train encoder and decoder')
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    return parser.parse_args()

def main(args, config):
    set_random_seed(42)
    base_folder = f'{args.base_folder}/{config["dataset"]}'
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(f'{base_folder}/exp', exist_ok=True)
    os.makedirs(f'{base_folder}/model', exist_ok=True)
    logger.info("Initializa data ...")

    logger.info("Coherence test start ...")
    dataset = load_homo_data(args.base_folder, config["dataset"], config["data_suffix"])
    if args.task=="link":
        train_data, _, _ = dataset.make_link_data(
            val_size=config[args.task]["val_size"], 
            test_size=config[args.task]["test_size"],
            seed=args.data_seed
        )
    else:
        dataset.make_classification_data(train_size=config[args.task]["train_size"])
        train_data = dataset.data
        
    semantic_relations = config["semantic_relation"][config['data_suffix']]
    for relation in semantic_relations:
        flow = HomoIntrusionTestFlow(
            base_folder=args.base_folder,
            dataset_name=config["dataset"],
            data_suffix=config["data_suffix"],
            topk = config["topk"],
            num_intruders = config["num_intruders"],
            num_buckets = config["num_buckets"],
            num_workers = config["num_workers"],
            num_processes = config["num_processes"],
        )

        flow.gen_test(
            relation=relation,
            sim_threshold=config["similarity_threshold"],
            train_data=train_data,
        )
    


if __name__ == '__main__':
    dist_mx = []
    logger = get_logger()
    args = parse_args()
    config = json.load(open(args.config_file, "r"))
    main(args, config)
