import os
import json
import pickle
import argparse
import torch

from exp_lib.utils import set_random_seed, get_logger


def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    parser.add_argument('--base-folder', type=str)
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--embed-dim', nargs="+", type=int)
    parser.add_argument('--data-suffix', type=str, default="baseline")
    return parser.parse_args()


def train_encoder(sim_mx):
    U, S, Vh = torch.linalg.svd(sim_mx)

    sorted_vidxs = torch.argsort(S, descending=True)
    sorted_U = U[:, sorted_vidxs]
    sorted_S = torch.sqrt(torch.diag(S[sorted_vidxs]))

    return sorted_U, sorted_S

def run():
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset_name = config["dataset"]
    os.makedirs(f"{args.base_folder}/{dataset_name}/svd", exist_ok=True)

    logger.info(f"SVD for {dataset_name}")
    for relation in config["semantic_relation"][args.data_suffix]:
        if relation[:-1]=="random":
            continue
        logger.info(f"Start decompose {relation}")
        sim_mx_file = f'{args.base_folder}/{dataset_name}/coherence_test/{relation}/{relation}.pkl'
        sim_mx = pickle.load(open(sim_mx_file, 'rb'))
        sim_mx = torch.Tensor(sim_mx)
        sorted_U, sorted_S = train_encoder(sim_mx)
        for embed_dim in args.embed_dim:
            if embed_dim==0:
                embed_dim=sim_mx.shape[0]
            z = sorted_U[:,:embed_dim] @ sorted_S[:embed_dim][:, :embed_dim]
            # z = torch.nn.functional.normalize(z)
            model_file = f"{args.base_folder}/{dataset_name}/model/exp_{dataset_name}_{args.data_suffix}_svd_{relation}_{embed_dim}.pt"
            torch.save(z.cpu(), model_file)
            logger.info(f"Output: {model_file}")
        

if __name__ == '__main__':
    logger = get_logger()
    args = parse_args()
    config = json.load(open(args.config_file, "r"))
    run()