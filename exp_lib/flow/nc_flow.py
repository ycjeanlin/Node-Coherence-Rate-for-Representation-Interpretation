import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score 
from torch_geometric.utils import coalesce
import torch.nn as nn

from exp_lib.model.decoder import LogReg, EnsembleMLP
from exp_lib.utils import get_logger
from .utils import EarlyStopping


logger = get_logger()

class NCFlow():
    def __init__(
        self, 
        graph_type,
        input_dim, 
        output_dim, 
        lr, 
        l2, 
        epoch,
        batch_size, 
        dropout
    ):
        self.graph_type = graph_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.l2 = l2
        self.epoch = epoch
        self.batch_size = batch_size
        self.dropout = dropout

    def run(self, z, data,  device, checkpoint_decoder_file, patience=20):
        decoder = self.init_classifier(self.input_dim, self.output_dim, self.dropout)
        decoder = decoder.to(device)
        decoder_optimizer = torch.optim.Adam(list(decoder.parameters()), lr=self.lr, weight_decay=self.l2)
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=checkpoint_decoder_file)
        for epoch in range(1, self.epoch+1):
            decoder_loss = self.train_classifier(z, data, decoder, decoder_optimizer)
            train_acc, val_acc, test_acc = self.test_classifier(z, data, decoder)
            if (epoch % 20)==0:
                logger.info(f'Epoch: {epoch:02d}, Decoder Loss: {decoder_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            early_stopping(decoder_loss, decoder)
            if early_stopping.early_stop:
                logger.debug('Early stopping!')
                break

        decoder = torch.load(checkpoint_decoder_file)
        train_acc, val_acc, test_acc = self.test_classifier(z, data, decoder)
        logger.info(f'Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        exp_df = self.output_classification_exp(z, data, decoder)
        return exp_df

    def init_classifier(self, input_dim, output_dim, dropout):
        decoder = LogReg(input_dim, output_dim, dropout)
        return decoder

    def train_classifier(self, z, data, decoder, optimizer):
        decoder.train()
        train_idx = data.train_mask.nonzero().cpu().numpy().reshape(-1)
        np.random.shuffle(train_idx)
        batch_size = self.batch_size 
        if batch_size==-1:
            batch_size=len(train_idx)
        losses = []
        for step, start in enumerate(range(0, len(train_idx), batch_size)):
            optimizer.zero_grad()
            batch_idxs = train_idx[start:start+batch_size]
            y_score = decoder(z[batch_idxs])
            loss = decoder.ce_loss(y_score, data.y[batch_idxs])
            losses.append(loss)
            loss.backward()
            optimizer.step()
        return torch.Tensor(losses).mean()


    def test_classifier(self, z, data, decoder):
        decoder.eval()
        if self.graph_type=="homo":
            y_true = data.y.cpu().numpy()
        elif self.graph_type=="hete":
            y_true = torch.zeros(data.train_mask.shape[0]).long()
            y_true[data.label_nodes.cpu()] = data.y.cpu()
            assert (y_true[data.train_mask.cpu()]>=0).all()
            assert (y_true[data.val_mask.cpu()]>=0).all()
            assert (y_true[data.test_mask.cpu()]>=0).all()
            y_true = y_true.cpu().numpy()
        else:
            raise ValueError('Incorrect graph type')
        with torch.no_grad():
            y_score = decoder.predict(z).detach().cpu().numpy()
            y_pred = np.argmax(y_score, axis=-1)
            train_acc = accuracy_score(y_true[data.train_mask.cpu().numpy()], y_pred[data.train_mask.cpu().numpy()])
            val_acc = accuracy_score(y_true[data.val_mask.cpu().numpy()], y_pred[data.val_mask.cpu().numpy()])
            test_acc = accuracy_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()])
        return train_acc, val_acc, test_acc

    def output_classification_exp(self, z, data, decoder):
        decoder.eval()
        with torch.no_grad():
            y_score = decoder.predict(z).detach()

        if self.graph_type=="homo":
            y_score = y_score.cpu().numpy()
            train_mask = data.train_mask.cpu().numpy()
            val_mask = data.val_mask.cpu().numpy()
            test_mask = data.test_mask.cpu().numpy()
        elif self.graph_type=="hete":
            y_score = y_score[data.label_nodes].cpu().numpy()
            train_mask = data.train_mask[data.label_nodes].cpu().numpy()
            val_mask = data.val_mask[data.label_nodes].cpu().numpy()
            test_mask = data.test_mask[data.label_nodes].cpu().numpy()
        else:
            raise ValueError('Incorrect graph type')
        
        exp_data = {}
        exp_data["y_true"] = data.y.cpu().numpy()
        exp_data["y_pred"] = np.argmax(y_score, axis=-1)
        for i in range(y_score.shape[1]):
            exp_data[f"y_score_{i}"] = y_score[:, i]
        exp_data["train_mask"] = train_mask
        exp_data["val_mask"] = val_mask 
        exp_data["test_mask"] = test_mask 
        df = pd.DataFrame.from_dict(exp_data)
        return df

class EnsembleNCFlow():
    def __init__(
        self, 
        graph_type,
        input_dims, 
        output_dim,
        hidden_dim, 
        lr, 
        l2, 
        epoch,
        batch_size, 
        dropout
    ):
        self.graph_type = graph_type
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.l2 = l2
        self.epoch = epoch
        self.batch_size = batch_size
        self.dropout = dropout

    def run(self, features, data,  device, checkpoint_decoder_file, patience=20):
        decoder = self.init_classifier(self.input_dims, self.output_dim, self.hidden_dim, self.dropout)
        decoder = decoder.to(device)
        decoder_optimizer = torch.optim.Adam(list(decoder.parameters()), lr=self.lr, weight_decay=self.l2)
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=checkpoint_decoder_file)
        for epoch in range(1, self.epoch+1):
            decoder_loss = self.train_classifier(features, data, decoder, decoder_optimizer)
            train_acc, val_acc, test_acc = self.test_classifier(features, data, decoder)
            if (epoch % 20)==0:
                logger.info(f'Epoch: {epoch:02d}, Decoder Loss: {decoder_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            early_stopping(decoder_loss, decoder)
            if early_stopping.early_stop:
                logger.debug('Early stopping!')
                break

        decoder = torch.load(checkpoint_decoder_file)
        train_acc, val_acc, test_acc = self.test_classifier(features, data, decoder)
        logger.info(f'Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        exp_df = self.output_classification_exp(features, data, decoder)
        return exp_df

    def init_classifier(self, input_dims, output_dim, hidden_dim, dropout):
        decoder = EnsembleMLP(input_dims, output_dim, hidden_dim, dropout)
        return decoder

    def train_classifier(self, features, data, decoder, optimizer):
        decoder.train()
        train_idx = data.train_mask.nonzero().cpu().numpy().reshape(-1)
        np.random.shuffle(train_idx)
        batch_size = self.batch_size 
        if batch_size==-1:
            batch_size=len(train_idx)
        losses = []
        for step, start in enumerate(range(0, len(train_idx), batch_size)):
            optimizer.zero_grad()
            batch_idxs = train_idx[start:start+batch_size]
            y_score = decoder(features, batch_idxs)
            loss = decoder.ce_loss(y_score, data.y[batch_idxs])
            losses.append(loss)
            loss.backward()
            optimizer.step()
        return torch.Tensor(losses).mean()


    def test_classifier(self, features, data, decoder):
        decoder.eval()
        if self.graph_type=="homo":
            y_true = data.y.cpu().numpy()
        elif self.graph_type=="hete":
            y_true = torch.zeros(data.train_mask.shape[0]).long()
            y_true[data.label_nodes.cpu()] = data.y.cpu()
            assert (y_true[data.train_mask.cpu()]>=0).all()
            assert (y_true[data.val_mask.cpu()]>=0).all()
            assert (y_true[data.test_mask.cpu()]>=0).all()
            y_true = y_true.cpu().numpy()
        else:
            raise ValueError('Incorrect graph type')
        all_idxs = np.arange(data.num_nodes)
        with torch.no_grad():
            y_score = decoder.predict(features, all_idxs).detach().cpu().numpy()
            y_pred = np.argmax(y_score, axis=-1)
            train_acc = accuracy_score(y_true[data.train_mask.cpu().numpy()], y_pred[data.train_mask.cpu().numpy()])
            val_acc = accuracy_score(y_true[data.val_mask.cpu().numpy()], y_pred[data.val_mask.cpu().numpy()])
            test_acc = accuracy_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()])
        return train_acc, val_acc, test_acc

    def output_classification_exp(self, features, data, decoder):
        decoder.eval()
        all_idxs = np.arange(data.num_nodes)
        with torch.no_grad():
            y_score = decoder.predict(features, all_idxs).detach()

        if self.graph_type=="homo":
            y_score = y_score.cpu().numpy()
            train_mask = data.train_mask.cpu().numpy()
            val_mask = data.val_mask.cpu().numpy()
            test_mask = data.test_mask.cpu().numpy()
        elif self.graph_type=="hete":
            y_score = y_score[data.label_nodes].cpu().numpy()
            train_mask = data.train_mask[data.label_nodes].cpu().numpy()
            val_mask = data.val_mask[data.label_nodes].cpu().numpy()
            test_mask = data.test_mask[data.label_nodes].cpu().numpy()
        else:
            raise ValueError('Incorrect graph type')
        
        exp_data = {}
        exp_data["y_true"] = data.y.cpu().numpy()
        exp_data["y_pred"] = np.argmax(y_score, axis=-1)
        for i in range(y_score.shape[1]):
            exp_data[f"y_score_{i}"] = y_score[:, i]
        exp_data["train_mask"] = train_mask
        exp_data["val_mask"] = val_mask 
        exp_data["test_mask"] = test_mask 
        df = pd.DataFrame.from_dict(exp_data)
        return df
