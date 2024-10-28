import json
import argparse
import torch

from exp_lib.utils import get_logger
from exp_lib.dataset import load_homo_data
from exp_lib.flow import LinkFlow 


def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    parser.add_argument('--task', type=str, default="link", help='method to train encoder and decoder')
    parser.add_argument('--base-folder', type=str)
    parser.add_argument('--model-suffix', type=str, default="baseline")
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--test-config', type=str)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    return parser.parse_args()

def train_model(dataset, train_data, val_data, test_data, embed_file, device):
    flow = LinkFlow()
    z = torch.load(embed_file)
    exp_df = flow.run(
        z,
        dataset.target_nodes,
        train_data,
        test_data,
        device,
    )
    flow.print_result(exp_df)
    return exp_df

def main(args, config):
    logger.info("Initializa data ...")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dataset_name = test_config["dataset"]
    data_suffix = test_config["data_suffix"]
    
    if config["graph_type"]=="homo":
        dataset = load_homo_data(args.base_folder, dataset_name, data_suffix)
        train_data, val_data, test_data = dataset.make_link_data(
            val_size=test_config[args.task]["val_size"],
            test_size=test_config[args.task]["test_size"], 
            seed=args.data_seed
        )
        embed_file = f'{args.base_folder}/{config["dataset"]}/model/{args.task}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}.pt'
    elif config["graph_type"]=="hete":
        dataset = load_hete_data(args.base_folder, dataset_name, data_suffix, test_config["edge_type"], test_config["rev_edge_type"])
        train_data, val_data, test_data = dataset.make_link_data(
            val_size=test_config[args.task]["val_size"],
            test_size=test_config[args.task]["test_size"], 
            seed=args.data_seed
        )
        source_type = dataset.edge_type[0]
        target_type = dataset.edge_type[2]
        embed_file = f'{args.base_folder}/{config["dataset"]}/model/{args.task}_{source_type}_{target_type}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}.pt'
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    logger.info("Model training start ...")
    exp_df = train_model(dataset, train_data, val_data, test_data, embed_file, device)

    if config["graph_type"]=="hete":
        source_type = dataset.edge_type[0]
        target_type = dataset.edge_type[2]
        out_exp_file = f'{args.base_folder}/{config["dataset"]}/exp/{task}_{source_type}_{target_type}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}_{args.data_seed}.csv'
        exp_df.to_csv(out_exp_file)
        logger.info(f"Output exp result to {out_exp_file}")
    elif config["graph_type"]=="homo":
        out_exp_file = f'{args.base_folder}/{config["dataset"]}/exp/{task}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}_{args.data_seed}.csv'
        exp_df.to_csv(out_exp_file)
        logger.info(f"Output exp result to {out_exp_file}")
        

if __name__ == '__main__':
    logger = get_logger()
    args = parse_args()
    task = "link"
    config = json.load(open(args.model_config, "r"))
    test_config = json.load(open(args.test_config, "r"))
    main(args, config)
