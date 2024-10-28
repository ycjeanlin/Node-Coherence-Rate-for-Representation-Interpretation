import os
import time
import json
import argparse
import torch

from exp_lib.utils import set_random_seed, get_logger, pass_rate_upper_bound, random_test
from exp_lib.dataset import load_homo_data
from exp_lib.flow import HomoIntrusionTestFlow

def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    parser.add_argument('--base-folder', type=str)
    parser.add_argument('--test-config-file', type=str)
    parser.add_argument('--model-config-file', type=str)
    parser.add_argument('--model-suffix', default="baseline", type=str)
    parser.add_argument('--task', type=str, help='method to train encoder and decoder')
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    return parser.parse_args()

def main():
    set_random_seed(42)
    base_folder = f'{args.base_folder}/{test_config["dataset"]}'
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(f'{base_folder}/exp', exist_ok=True)
    os.makedirs(f'{base_folder}/model', exist_ok=True)
    logger.info("Initializa data ...")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if test_config["graph_type"]=="hete":
        source_type = test_config["edge_type"][0]
        target_type = test_config["edge_type"][2]
        embed_file = f'{args.base_folder}/{test_config["dataset"]}/model/{args.task}_{source_type}_{target_type}_{test_config["dataset"]}_{test_config["data_suffix"]}_{model_config["model"]}_{args.model_suffix}.pt'
    else:
        embed_file = f'{args.base_folder}/{test_config["dataset"]}/model/{args.task}_{test_config["dataset"]}_{test_config["data_suffix"]}_{model_config["model"]}_{args.model_suffix}.pt'


    logger.info("Coherence test start ...")
    dataset = load_homo_data(
        args.base_folder, 
        test_config["dataset"],
        test_config["data_suffix"],
    )
    if args.task=="link":
        train_data, _, _ = dataset.make_link_data(
            val_size=test_config[args.task]["val_size"], 
            test_size=test_config[args.task]["test_size"],
            seed=args.data_seed
        )
        train_data = train_data.to(device)
    else:
        dataset.make_classification_data(train_size=test_config[args.task]["train_size"])
        train_data = dataset.data.to(device)

    if model_config["model"]!="random":
        z = torch.load(embed_file)
    else:
        z = torch.rand(train_data.num_nodes, 128)

    semantic_relations = test_config["semantic_relation"][test_config['data_suffix']]
    epsilon, ub_intrusion_pass_rate, ub_ranking_pass_rate = random_test(
        z, 
        test_config["topk"], 
        test_config["num_intruders"], 
        test_config["intrusion_decision_quantile"], 
    )
    
    intrusion_ub = pass_rate_upper_bound(z, test_config["topk"], epsilon)
    flow = HomoIntrusionTestFlow(
        base_folder=args.base_folder,
        dataset_name=test_config["dataset"],
        data_suffix=test_config["data_suffix"],
        topk = test_config["topk"],
        num_intruders = test_config["num_intruders"],
        num_buckets = test_config["num_buckets"],
        num_workers = test_config["num_workers"],
        num_processes = test_config["num_processes"],
    )

    for relation in semantic_relations:
        logger.info(f"Coherence test for {relation} ...")
        exp_df = flow.run_test(
            relation=relation,
            model_name=model_config["model"],
            node_embed=z,
            coherence_epsilon=epsilon,
        )

        exp_df["Cantelli Upper Bound"]=[intrusion_ub] * len(exp_df)
        exp_df["Intrusion Upper Bound"]=[ub_intrusion_pass_rate] * len(exp_df)
        exp_df["Ranking Upper Bound"]=[ub_ranking_pass_rate] * len(exp_df)
        out_test_file = f"{args.base_folder}/{test_config['dataset']}/exp/{relation}_{args.task}_{test_config['dataset']}_{test_config['data_suffix']}_{model_config['model']}_{args.model_suffix}.csv"
        exp_df.to_csv(out_test_file)
        logger.info(f"Output Exp File: {out_test_file}")
            
    
if __name__ == '__main__':
    dist_mx = []
    logger = get_logger()
    args = parse_args()
    test_config = json.load(open(args.test_config_file, "r"))
    model_config = json.load(open(args.model_config_file, "r"))
    main()
