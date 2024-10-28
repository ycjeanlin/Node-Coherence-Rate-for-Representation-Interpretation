import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score 
from torch_geometric.utils import coalesce

from sklearn.metrics import accuracy_score

from exp_lib.model.dgcn import DGCN
from exp_lib.model.decoder import LinkDecoder
from exp_lib.utils import get_logger, cal_rank

from .utils import EarlyStopping

logger = get_logger()

class DGCNFlow():
    def __init__(self, 
        input_dim,
        embed_dim, 
        with_feat, 
        epoch, 
        encoder_lr, 
        encoder_l2, 
        encoder_batch_size, 
        dropout,
        num_layers,
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layer = num_layers
        self.dropout = dropout
        self.with_feat = with_feat
        self.encoder_lr = encoder_lr
        self.encoder_l2 = encoder_l2
        self.epoch = epoch
        self.batch_size = encoder_batch_size

    def run(self, x, train_data, device, checkpoint_embed_file, patience=20):
        encoder = self.init_model(
            train_data, 
            self.input_dim, 
            self.embed_dim, 
            self.embed_dim, 
            self.num_layer,
            self.with_feat,
            self.dropout,
        )
        encoder = encoder.to(device)
        encoder_optimizer = torch.optim.Adam(list(encoder.parameters()), lr=self.encoder_lr, weight_decay=self.encoder_l2)
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=checkpoint_embed_file)
        for epoch in range(1, self.epoch+1):
            encoder_loss, z = self.train_encoder(x, train_data, encoder, encoder_optimizer)
            encoder.eval()
            z = encoder.embed(x, train_data.edge_index)
            if epoch==1 or (epoch%20)==0:
                logger.info(f'Epoch: {epoch:02d}, Encoder Loss: {encoder_loss:.4f}')
            early_stopping(encoder_loss, z)
            if early_stopping.early_stop:
                logger.info(f'Epoch: {epoch:02d}, Encoder Loss: {encoder_loss:.4f}')
                break

        z = torch.load(checkpoint_embed_file)
        return z

    def init_model(self, data, input_dim, out_dim, embed_dim, num_layers, with_feat, dropout):
        encoder = DGCN(
            input_dim, out_dim, embed_dim, num_layers,
            data.num_nodes, with_feat, p=dropout
        )
        return encoder

    def train_encoder(self, x, data, encoder, optimizer):
        encoder.train()  
        train_idx = np.arange(data.edge_index.shape[1])
        np.random.shuffle(train_idx)
        batch_size = self.batch_size
        if batch_size==-1:
            batch_size = len(train_idx)
        losses = []
        for step, start in enumerate(range(0, len(train_idx), batch_size)):
            optimizer.zero_grad()
            z = encoder(x, data.edge_index)
            batch_edge_index = data.edge_index[:, train_idx[start:start+batch_size]]
            loss = encoder.recon_loss(z, batch_edge_index)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        return  torch.Tensor(losses).mean(), z

    def output_embed(self, x, train_pos_edge_index, encoder):
        encoder.eval()
        with torch.no_grad():
            z = encoder(x, train_pos_edge_index)
        return z
    
    

class DGCNSupFlow():
    
    def __init__(self, 
        input_dim, 
        output_dim, 
        embed_dim, 
        with_feat, 
        epoch, 
        encoder_lr, 
        encoder_l2, 
        dropout, 
        num_layer, 
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        self.with_feat = with_feat
        self.encoder_lr = encoder_lr
        self.encoder_l2 = encoder_l2
        self.epoch = epoch
        self.dropout = dropout

    def run_classification(self, x, train_data, device, checkpoint_model_file, patience=20):
        encoder = self.init_model(
            train_data, 
            self.input_dim, 
            self.output_dim, 
            self.embed_dim, 
            self.num_layer,
            self.with_feat,
            self.dropout
        )
        encoder = encoder.to(device)
        encoder_optimizer = torch.optim.Adam(list(encoder.parameters()), lr=self.encoder_lr, weight_decay=self.encoder_l2)
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=checkpoint_model_file)
        for epoch in range(1, self.epoch+1):
            encoder_loss = self.train_encoder(x, train_data, encoder, encoder_optimizer)
            train_acc, val_acc, test_acc = self.test_classifier(x, train_data, encoder)
            if (epoch % 20)==0:
                logger.info(f'Epoch: {epoch:02d}, Loss: {encoder_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            early_stopping(encoder_loss, encoder)
            if early_stopping.early_stop:
                break

        encoder = torch.load(checkpoint_model_file)
        train_acc, val_acc, test_acc = self.test_classifier(x, train_data, encoder)
        logger.info(f'Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        exp_df = self.output_classification_exp(x, train_data, encoder)
        return exp_df

    def train_encoder(self, x, data, encoder, optimizer):
        encoder.train()
        ce_loss = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        z = encoder(x, data.edge_index)
        # z = torch.nn.functional.normalize(z)
        loss = ce_loss(z[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        return loss.item()

    def test_classifier(self, x, data, encoder):
        encoder.eval()
        with torch.no_grad():
            y_score = encoder.predict(x, data.edge_index).detach().cpu().numpy()
            # y_true = np.eye(len(data.y.unique()))[data.y.cpu()]
            y_true = data.y.cpu().numpy()
            y_pred = np.argmax(y_score, axis=-1)
            train_mask = data.train_mask.cpu().numpy()
            val_mask = data.val_mask.cpu().numpy()
            test_mask = data.test_mask.cpu().numpy()
            train_acc = accuracy_score(y_true[train_mask], y_pred[train_mask])
            val_acc = accuracy_score(y_true[val_mask], y_pred[val_mask])
            test_acc = accuracy_score(y_true[test_mask], y_pred[test_mask])
        return train_acc, val_acc, test_acc

    def output_classification_exp(self, x, data, decoder):
        decoder.eval()
        with torch.no_grad():
            y_score = decoder.predict(x, data.edge_index).detach().cpu().numpy()
        
        exp_data = {}
        exp_data["y_true"] = data.y.cpu().numpy()
        exp_data["y_pred"] = np.argmax(y_score, axis=-1)
        for i in range(y_score.shape[1]):
            exp_data[f"y_score_{i}"] = y_score[:, i]
        exp_data["train_mask"] = data.train_mask.cpu().numpy() 
        exp_data["val_mask"] = data.val_mask.cpu().numpy() 
        exp_data["test_mask"] = data.test_mask.cpu().numpy() 
        df = pd.DataFrame.from_dict(exp_data)
        return df

    def init_model(self, data, input_dim, out_dim, embed_dim, num_layers, with_feat, dropout):
        encoder = DGCN(
            input_dim, out_dim, embed_dim, num_layers,
            data.num_nodes, with_feat, p=dropout
        )
        return encoder


    def output_embed(self, x, train_pos_edge_index, encoder):
        encoder.eval()
        with torch.no_grad():
            z = encoder(x, train_pos_edge_index)
        return z