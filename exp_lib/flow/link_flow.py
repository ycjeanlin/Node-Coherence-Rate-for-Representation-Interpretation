import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score 
from torch_geometric.utils import coalesce

from exp_lib.model.decoder import LinkDecoder
from exp_lib.utils import get_logger, cal_rank, hit_rate, mrr


logger = get_logger()

class LinkFlow():
    def __init__(self):
        pass

    def run(self, z, target_nodes, train_data, test_data, device):
        decoder = self.init_model()
        decoder = decoder.to(device)
        exp_df = self.output_exp(z, train_data, test_data, target_nodes, decoder)
        
        return exp_df 

    def init_model(self):
        decoder = LinkDecoder()
        return decoder

    def output_exp(self, z, train_data, test_data, target_nodes, decoder):
        decoder.eval()
        with torch.no_grad():
            train_score = decoder(z, train_data.edge_label_index, sigmoid=True).detach().cpu()
            test_score = decoder(z, test_data.edge_label_index, sigmoid=True).detach().cpu()
            edge_index = torch.cat((train_data.edge_label_index, test_data.edge_label_index), dim=-1).cpu()
            edge_label = torch.cat((train_data.edge_label, test_data.edge_label), dim=-1).cpu()
            y_score= torch.cat((train_score, test_score), dim=-1)
            train_mask = torch.zeros(edge_index.shape[1], dtype=bool)
            train_mask[:train_data.edge_label_index.shape[1]] = True
            edge_index, edge_attr = coalesce(edge_index, edge_attr=[train_mask, edge_label, y_score])
            ranks = cal_rank(z, edge_index, target_nodes)

        exp_data = {}
        train_mask = edge_attr[0].numpy()
        test_mask = (~train_mask)
        exp_data["row"] = edge_index[0].cpu().numpy()
        exp_data["col"] = edge_index[1].cpu().numpy()
        exp_data["rank"] = ranks.cpu().numpy()
        exp_data["y_true"] = edge_attr[1].cpu().numpy()
        exp_data["y_score"] = edge_attr[2].cpu().numpy()
        exp_data["train_mask"] = train_mask
        exp_data["test_mask"] = test_mask
        df = pd.DataFrame.from_dict(exp_data)
        return df

    def print_result(self, df):
        test_mask = df["test_mask"]
        edge_mask = (test_mask & (df["y_true"]==1))
        y_score = df["y_score"][test_mask]
        y_true = df["y_true"][test_mask]
        rank = torch.Tensor(df["rank"][edge_mask].values)
        row_index = torch.LongTensor(df["row"][edge_mask].values)
        auc = roc_auc_score(y_true, y_score)
        rank_score = mrr(rank, row_index)
        hit_k5 = hit_rate(rank, row_index, topk=5)
        hit_k10 = hit_rate(rank, row_index, topk=10)
        logger.info(f"AUC: {auc:.3f}, MRR: {rank_score:.3f}, HIT@5: {hit_k5:.3f}, HIT@10: {hit_k10:.3f}")
        