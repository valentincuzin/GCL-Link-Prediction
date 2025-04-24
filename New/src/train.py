from functools import partial
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
import numpy as np


@torch.no_grad()
def test(encoder: nn.Module, predictor: nn.Module, data, split_edge: dict, hp: dict):
    if isinstance(encoder, nn.Module):
        encoder.eval()
    if isinstance(predictor, nn.Module):
        predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = None if encoder is None else encoder(data.x, adj_t)
    def test_split(split):
        # pred positive edges and negatives edges for nodes in the split
        pos_test_edge = split_edge[split]['edge'].to(device)
        neg_test_edge = split_edge[split]['edge_neg'].to(device)
        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), hp['batch_size']):
            # print('perm', perm)
            edge = pos_test_edge[perm].t()
            # print('edge', edge)
            out = predictor.predict(h, edge[0], edge[1])
            pos_test_preds += [out.squeeze().cpu()]
        # print(pos_test_preds)
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), hp['batch_size']):
            edge = neg_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1])
            neg_test_preds += [out.squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        return pos_test_pred, neg_test_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')
    if hp['use_valedges_as_input']:
        adj_t = data.full_adj_t
        h = None if encoder is None else encoder(data.x, adj_t)
    pos_test_pred, neg_test_pred = test_split('test')
    # print("pos_test_pred", pos_test_pred)
    return pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred

def pred_train(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          loss_name: str,
          hp: dict):
    if not hp['freeze']:
        optimizer = torch.optim.Adam(
            [
                {"params": encoder.parameters(), "lr": hp["gnn_lr"]},
                {"params": predictor.parameters(), "lr": hp["pre_lr"]},
            ]
        )
    elif hp['freeze']:
        optimizer = torch.optim.Adam(params=predictor.parameters(), lr=hp["pre_lr"])
    encoder.train()
    predictor.train()
    return _train(encoder, predictor, data, split_edge, optimizer, hp, loss_name)

def baseline_train(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          loss_name: str,
          hp: dict):
    if isinstance(predictor, nn.Module):
        predictor = predictor.to(data.x.device)
        optimizer = torch.optim.Adam(
            [
                {"params": encoder.parameters(), "lr": hp["gnn_lr"],  "weight_decay": hp['weight_decay']},
                {"params": predictor.parameters(), "lr": hp["pre_lr"]},
            ]
        )
        predictor.train()
    else:
        optimizer = torch.optim.Adam(params=encoder.parameters(), lr=hp["gnn_lr"], weight_decay=hp['weight_decay'])
    encoder.train()
    return _train(encoder, predictor, data, split_edge, optimizer, hp, loss_name)

def _train(encoder, predictor, data, split_edge, optimizer, hp, loss_name):
    loss_res = []
    t1 = time.time()
    device = data.adj_t.device()

    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    for epoch in tqdm(range(1, 1 + hp["epochs"])):
        negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
        for perm in DataLoader(range(adjmask.size(0)), hp['batch_size'], shuffle=True):
            optimizer.zero_grad()
            if hp['mask_input']:
                adjmask[perm] = 0
                tei = pos_train_edge[:, adjmask]
                adj = SparseTensor.from_edge_index(tei,
                                sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                    pos_train_edge.device, non_blocking=True)
                adjmask[perm] = 1
                adj = adj.to_symmetric()
            else:
                adj = data.adj_t
            h = encoder(data.x, adj)
            edge = pos_train_edge[:, perm]
            pos_outs = predictor(h, edge[0], edge[1])

            edge = negedge[:, perm]
            neg_outs = predictor(h, edge[0], edge[1])
            loss = get_loss(loss_name, pos_outs, neg_outs)
            loss.backward()
            optimizer.step()
            total_loss.append(loss)
        if epoch % 10 == 0:
            _loss = np.average([_.item() for _ in total_loss])
            loss_res.append(round(float(_loss), 2))
    print('train loss: ', loss_res)
    print(f"train time: {time.time()-t1:.2f} s")
    return total_loss

def get_loss(loss_name: str, pos_outs, neg_outs):
    switch = {
        'log_sig': partial(log_sig_loss, pos_outs, neg_outs),
        'bce': partial(bce_loss, pos_outs, neg_outs),
        'auc': partial(auc_loss, pos_outs, neg_outs),
        'hinge_auc': partial(hinge_auc_loss, pos_outs, neg_outs),
    }
    return switch[loss_name]()

def log_sig_loss(pos_outs, neg_outs):
    pos_losss = -F.logsigmoid(pos_outs).mean()
    neg_losss = -F.logsigmoid(-neg_outs).mean()
    return neg_losss + pos_losss

def bce_loss(pos_out, neg_out):
    out = torch.cat((pos_out, neg_out), dim=0).to(pos_out.device)
    label = torch.cat((torch.ones(pos_out.size()), torch.zeros(neg_out.size())), dim=0).to(pos_out.device)
    return F.binary_cross_entropy_with_logits(out, label, reduction="mean")

def auc_loss(pos_out, neg_out):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, 1))
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, 1))
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

