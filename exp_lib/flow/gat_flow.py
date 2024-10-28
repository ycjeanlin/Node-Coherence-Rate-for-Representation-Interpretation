import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from exp_lib.model.gat import GAT
from exp_lib.utils import get_logger, set_random_seed
from .utils import EarlyStopping


logger = get_logger()

class GATFlow():
    def __init__(
        self, 
        input_dim, 
        embed_dim, 
        with_feat, 
        epoch,
        encoder_lr, 
        encoder_l2, 
        encoder_batch_size, 
        encoder_dropout, 
        num_layers, 
        num_heads, 
        pretrain_file=None
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.encoder_dropout = encoder_dropout
        self.with_feat = with_feat
        self.with_target = False
        self.encoder_lr = encoder_lr
        self.encoder_l2 = encoder_l2
        self.epoch = epoch
        self.encoder_batch_size = encoder_batch_size
        self.pretrain_file = pretrain_file

    def run(self, 
        x, 
        data, 
        device, 
        checkpoint_embed_file, 
        patience
    ):
        encoder = self.init_encoder(
            data, 
            self.input_dim, 
            self.embed_dim, 
            self.embed_dim, 
            self.num_layers, 
            self.num_heads, 
            self.with_feat,
            self.with_target,
            self.encoder_dropout
        )
        encoder = encoder.to(device)
        encoder_optimizer = torch.optim.Adam(list(encoder.parameters()), lr=self.encoder_lr, weight_decay=self.encoder_l2)
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=checkpoint_embed_file)
        for epoch in range(1, self.epoch+1):
            encoder_loss, z = self.train_encoder(x, data, encoder, encoder_optimizer)
            encoder.eval()
            z = encoder.embed(x, data.edge_index)
            if (epoch % 20)==0 or epoch==1:
                logger.info(f'Epoch: {epoch:02d}, Encoder Loss: {encoder_loss:.4f}')
            early_stopping(encoder_loss, z.detach())
            if early_stopping.early_stop:
                logger.info(f'Epoch: {epoch:02d}, Encoder Loss: {encoder_loss:.4f}')
                break
        
        z = torch.load(checkpoint_embed_file)
        return z

    def init_encoder(self, data, input_dim, out_dim, embed_dim, num_layers, num_heads,  with_feat, with_target, dropout=0.5):
        encoder = GAT(
            input_dim, out_dim, embed_dim, num_layers, num_heads,   
            data.num_nodes, with_feat=with_feat, with_target=with_target, p=dropout,
            pretrain_file=self.pretrain_file
        )

        return encoder

    def train_encoder(self, x, data, encoder, optimizer):
        encoder.train()
        train_idx = np.arange(data.edge_index.shape[1])
        np.random.shuffle(train_idx)
        if self.encoder_batch_size==-1:
            batch_size = len(train_idx)
        else:
            batch_size = self.encoder_batch_size 
        losses = []
        for step, start in enumerate(range(0, len(train_idx), batch_size)):
            # if len(train_idx)<start+batch_size:
            #     continue
            optimizer.zero_grad()
            z = encoder(x, data.edge_index)
            batch_edge_index = data.edge_index[:, train_idx[start:start+batch_size]]
            loss = encoder.recon_loss(z, batch_edge_index)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        return torch.Tensor(losses).mean(), z

    def output_embed(self, x, train_pos_edge_index, encoder):
        encoder.eval()
        with torch.no_grad():
            z = encoder(x, train_pos_edge_index)
        return z


class GATSupFlow():
    def __init__(self, 
        input_dim,
        output_dim, 
        embed_dim, 
        with_feat, 
        epoch, 
        encoder_lr, 
        encoder_l2, 
        encoder_batch_size, 
        dropout,
        num_layers, 
        num_heads, 
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.with_feat = with_feat
        self.with_target = False
        self.encoder_lr = encoder_lr
        self.encoder_l2 = encoder_l2
        self.epoch = epoch
        self.encoder_batch_size = encoder_batch_size
        self.dropout = dropout

    def run_classification(self, x, data, device, checkpoint_model_file, patience):
        encoder = self.init_encoder(
            data, 
            self.input_dim, 
            self.output_dim, 
            self.embed_dim, 
            self.num_layers, 
            self.num_heads, 
            self.with_feat,
            self.with_target,
            self.dropout
        )
        encoder = encoder.to(device)
        encoder_optimizer = torch.optim.Adam(list(encoder.parameters()), lr=self.encoder_lr, weight_decay=self.encoder_l2)
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=checkpoint_model_file)

        for epoch in range(1, self.epoch+1):
            encoder_loss = self.train_encoder(x, data, encoder, encoder_optimizer)
            train_acc, val_acc, test_acc = self.test_classifier(x, data, encoder)
            if (epoch % 20)==0 or epoch==1:
                logger.info(f'Epoch: {epoch:02d}, Loss: {encoder_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            encoder.eval()
            early_stopping(-val_acc, encoder)
            if early_stopping.early_stop:
                break
        
        encoder = torch.load(checkpoint_model_file)
        train_acc, val_acc, test_acc = self.test_classifier(x, data, encoder)
        logger.info(f'Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        exp_df = self.output_classification_exp(x, data, encoder)
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

    def init_encoder(self, data, input_dim, out_dim, embed_dim, num_layers, num_heads, with_feat, with_target, dropout=0.5):
        encoder = GAT(
            input_dim, out_dim, embed_dim, num_layers, num_heads,   
            data.num_nodes, with_feat=with_feat, with_target=with_target, p=dropout,
        )
        return encoder

    def output_embed(self, x, train_pos_edge_index, encoder):
        encoder.eval()
        with torch.no_grad():
            z = encoder(x, train_pos_edge_index)
        return z
    