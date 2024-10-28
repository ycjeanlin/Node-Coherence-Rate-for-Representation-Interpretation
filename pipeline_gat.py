import os 
import json
import argparse
import numpy as np
import pandas as pd
import torch

from exp_lib.utils import set_random_seed, get_logger
from exp_lib.dataset import load_homo_data, load_hete_data
from exp_lib.flow import GATFlow


def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    parser.add_argument('--base-folder', type=str)
    parser.add_argument('--model-suffix', type=str, default="baseline")
    parser.add_argument('--task', type=str, default="nc", help='method to train encoder and decoder')
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--test-config', type=str)
    parser.add_argument('--with-feat', action='store_true', help='Have node features')
    parser.add_argument('--sparse', action='store_true', help='Create sparse embedding')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--data-seed', type=int, default=42)
    return parser.parse_args()

def train_model(args, config, data, device, checkpoint_embed_file):
    if args.with_feat:
        input_dim = data.x.shape[-1]
        embed_dim = config[args.task]["encoder"]["embed_dim"]
        x = data.x
    else:
        input_dim = config[args.task]["encoder"]["embed_dim"]
        embed_dim = config[args.task]["encoder"]["embed_dim"]
        x = torch.arange(data.num_nodes).to(device)

    flow = GATFlow(
        input_dim, 
        embed_dim, 
        args.with_feat, 
        config[args.task]["encoder"]["epoch"], 
        config[args.task]["encoder"]["learning_rate"], 
        config[args.task]["encoder"]["l2"], 
        config[args.task]["encoder"]["batch_size"], 
        config[args.task]["encoder"]["dropout"], 
        config[args.task]["encoder"]["num_layer"], 
        config[args.task]["encoder"]["num_head"], 
    )

    z = flow.run(
        x, 
        data,
        device,
        checkpoint_embed_file,
        30 
    )
    return z

def main(args, config):
    logger.info("Initializa data ...")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'{args.base_folder}/{config["dataset"]}/model', exist_ok=True)
    dataset_name = test_config["dataset"]
    data_suffix = test_config["data_suffix"]

    if config["graph_type"]=="homo":
        dataset = load_homo_data(args.base_folder, dataset_name, data_suffix)
        if args.task=="link":
            train_data, _, _ = dataset.make_link_data(
                val_size=test_config[args.task]["val_size"],
                test_size=test_config[args.task]["test_size"], 
                seed=args.data_seed
            )
        elif args.task=="nc":
            train_data = dataset.data
        out_embed_file = f'{args.base_folder}/{config["dataset"]}/model/{args.task}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}.pt'
    elif config["graph_type"]=="hete":
        dataset = load_hete_data(args.base_folder, dataset_name, data_suffix, test_config["edge_type"], test_config["rev_edge_type"])
        if args.task=="link":
            train_data, _, _ = dataset.make_link_data(
                val_size=test_config[args.task]["val_size"],
                test_size=test_config[args.task]["test_size"], 
                seed=args.data_seed
            )
        elif args.task=="nc":
            train_data = dataset.data
        source_type = dataset.edge_type[0]
        target_type = dataset.edge_type[2]
        out_embed_file = f'{args.base_folder}/{config["dataset"]}/model/{args.task}_{source_type}_{target_type}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}.pt'

    train_data = train_data.to(device)
    logger.info("Model training start ...")
    z = train_model(
        args, 
        config, 
        train_data, 
        device, 
        out_embed_file,
    )
    torch.save(z, out_embed_file)
    logger.info(f"Output model to {out_embed_file}")


if __name__ == '__main__':
    logger = get_logger()
    args = parse_args()
    config = json.load(open(args.model_config, "r"))
    test_config = json.load(open(args.test_config, "r"))
    main(args, config)
