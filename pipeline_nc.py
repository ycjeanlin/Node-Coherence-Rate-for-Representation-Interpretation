import json
import argparse
import torch

from exp_lib.utils import get_logger
from exp_lib.dataset import load_homo_data
from exp_lib.flow import NCFlow 


def parse_args():
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    parser.add_argument('--base-folder', type=str)
    parser.add_argument('--task', type=str, default="nc")
    parser.add_argument('--model-suffix', type=str, default="baseline")
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--test-config', type=str)
    parser.add_argument('--train-ratio', type=float, default=0.05)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    return parser.parse_args()

def train_model(args, config, train_data, embed_file, device):
    num_classes = len(torch.bincount(train_data.y))
    decoder_file = f'{args.base_folder}/{config["dataset"]}/model/{task}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_decoder.pt'
    flow = NCFlow(
        graph_type=test_config["graph_type"],
        input_dim=config[task]["encoder"]["embed_dim"],
        output_dim=num_classes,
        lr=config[task]["decoder"]["learning_rate"],
        l2=config[task]["decoder"]["l2"],
        epoch=config[task]["decoder"]["epoch"],
        batch_size=config[task]["decoder"]["batch_size"],
        dropout=config[task]["decoder"]["dropout"],
    )
    z = torch.load(embed_file).to(device)
    exp_df = flow.run(
        z,
        train_data,
        device,
        decoder_file,
        patience=30,
    )

    return exp_df

def main(args, config):
    logger.info("Initializa data ...")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    logger.info(f"SEED {args.data_seed}")
    dataset_name = test_config["dataset"]
    data_suffix = test_config["data_suffix"]

    
    dataset = load_homo_data(args.base_folder, dataset_name, data_suffix)
    if args.task=="link":
        train_data, _, _ = dataset.make_link_data(val_size=config[args.task]["val_size"],test_size=config[args.task]["test_size"], seed=args.data_seed)
    elif args.task=="nc":
        dataset.make_classification_data(train_size=test_config[args.task]["train_size"])
        train_data = dataset.data
    embed_file = f'{args.base_folder}/{config["dataset"]}/model/{args.task}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}.pt'

    train_data = dataset.data
    train_data = train_data.to(device)

    logger.info(f"Load embeddings: {embed_file}")
    logger.info("Model training start ...")
    exp_df = train_model(args, config, train_data, embed_file, device)

    out_exp_file = f'{args.base_folder}/{config["dataset"]}/exp/{task}_{config["dataset"]}_{config["data_suffix"]}_{config["model"]}_{args.model_suffix}.csv'
    exp_df.to_csv(out_exp_file)
    logger.info(f"Output exp result to {out_exp_file}")
        

if __name__ == '__main__':
    logger = get_logger()
    args = parse_args()
    task = "nc"
    config = json.load(open(args.model_config, "r"))
    test_config = json.load(open(args.test_config, "r"))
    main(args, config)
